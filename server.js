/**
 * server.js
 * Free-tier backend: Pexels photos + Groq vision reasoning -> build plan.
 */

require("dotenv").config();
const express = require("express");
const fetch = require("node-fetch");

const app = express();
app.use(express.json({ limit: "10kb" }));

const PEXELS_API_KEY = process.env.PEXELS_API_KEY;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GROQ_VISION_MODEL = "qwen/qwen3.6-27b";
const PORT = process.env.PORT || 3000;

// Groq's rate limits are per-organization, not per visitor, and a single
// build can use nearly the whole per-minute output budget by itself -- so
// this needs to be one global gate, spaced to actually clear each minute
// window, not a lenient per-IP check.
let lastGlobalRequestAt = 0;
const MIN_INTERVAL_MS = 65_000;
function isRateLimited() {
  const now = Date.now();
  if (now - lastGlobalRequestAt < MIN_INTERVAL_MS) return true;
  lastGlobalRequestAt = now;
  return false;
}

const FILLER_WORDS = [
  "a", "an", "the", "small", "big", "large", "tiny", "huge", "giant",
  "mini", "little", "enormous", "massive", "cozy", "compact",
  "red", "blue", "green", "yellow", "orange", "purple", "pink",
  "black", "white", "brown", "gray", "grey", "silver", "gold", "golden",
];

function simplifyForImageSearch(prompt) {
  const words = prompt.toLowerCase().split(/\s+/).filter((w) => w.length > 0 && !FILLER_WORDS.includes(w));
  return words.length > 0 ? words.join(" ") : prompt;
}

async function searchImages(query) {
  if (!PEXELS_API_KEY) return [];
  try {
    const res = await fetch(
      `https://api.pexels.com/v1/search?query=${encodeURIComponent(query)}&per_page=1`,
      { headers: { Authorization: PEXELS_API_KEY.trim() } }
    );
    if (!res.ok) return [];
    const data = await res.json();
    // "small" (~130px) instead of "medium" -- much smaller payload, so it
    // doesn't blow the model's input-tokens-per-minute budget.
    return (data.photos || []).map((p) => p.src.small);
  } catch (err) {
    console.error("Pexels search failed:", err.message);
    return [];
  }
}

async function fetchImageAsBase64(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const contentType = res.headers.get("content-type") || "image/jpeg";
    const buffer = await res.buffer();
    return { mimeType: contentType, base64: buffer.toString("base64") };
  } catch (err) {
    console.error("Image fetch failed:", err.message);
    return null;
  }
}

const SYSTEM_PROMPT = `You are a 3D building planner for Roblox. You will be shown a reference photo of a real object and must convert it into a JSON list of parts approximating its structure, proportions, and colors.

Respond with ONLY a single valid JSON object, nothing else. Base schema (use exactly these 7 keys per object, unless adding an optional extra described below):
{
  "objects": [
    {
      "class": "Part" | "WedgePart" | "CornerWedgePart" | "TrussPart" | "Seat" | "VehicleSeat",
      "shape": "Block" | "Ball" | "Cylinder",
      "size": [x, y, z],
      "position": [x, y, z],
      "color": [r, g, b],
      "material": "Plastic" | "Wood" | "Brick" | "Concrete" | "Metal" | "Glass" | "Grass" | "Fabric",
      "name": "short label, max 16 chars"
    }
  ],
  "notes": "under 12 words describing what you built"
}

Optional extras (only add these keys on the specific objects that need them -- most objects should NOT have them):
- "meshType": one of "Wedge" | "CornerWedge" | "Prism" | "Pyramid" | "ParallelRamp" | "RightAngleRamp" | "Torso" | "Head" | "Brick" -- gives a Part a more specific shape than plain Block/Ball/Cylinder.
- "light": { "type": "Point" | "Spot", "color": [r,g,b], "brightness": 1-5 } -- makes this part glow (windows, lamps, fires, headlights).

Rules:
- Base shapes/proportions on what's actually visible in the reference photo.
- Use 6-10 objects for anything more complex than a single simple item. Build a genuinely complete, recognizable structure -- all major sections/features -- never just one block. This has a strict output budget: stay at or under 10 objects, use short names, and add optional keys sparingly.
- Keep content appropriate for a general, all-ages audience.
- Your entire reply must be exactly one JSON object and nothing else -- no prose, no markdown fences.`;

// If the response got cut off mid-generation (hit the token limit), salvage
// whichever individual objects finished completely rather than throwing
// the whole build away. Matches top-level {...} blocks, allowing at most
// one level of nesting (for the optional "light" sub-object).
function salvageObjects(text) {
  const pattern = /\{(?:[^{}]|\{[^{}]*\})*\}/g;
  const matches = text.match(pattern) || [];
  const salvaged = [];
  for (const m of matches) {
    try {
      const obj = JSON.parse(m);
      if (obj && typeof obj === "object" && obj.class && obj.size && obj.position) {
        salvaged.push(obj);
      }
    } catch (e) {
      // incomplete trailing object -- skip it
    }
  }
  return salvaged;
}

function extractJson(text) {
  let cleaned = text.replace(/```json/gi, "").replace(/```/g, "").trim();

  const start = cleaned.indexOf("{");
  if (start !== -1) cleaned = cleaned.slice(start);

  const end = cleaned.lastIndexOf("}");
  const fullSlice = end !== -1 ? cleaned.slice(0, end + 1) : cleaned;

  const repaired = fullSlice.replace(/,\s*([\]}])/g, "$1").replace(/\/\/[^\n]*/g, "");

  try {
    return JSON.parse(repaired);
  } catch (e) {
    console.error("Raw model output that failed to parse:\n" + text.slice(0, 2000));
    const salvaged = salvageObjects(cleaned);
    if (salvaged.length > 0) {
      console.log(`Response was cut off -- salvaged ${salvaged.length} complete object(s) out of it.`);
      return { objects: salvaged, notes: "(partial build -- response was cut off)" };
    }
    throw new Error("Model output was not valid JSON even after repair: " + e.message);
  }
}

async function generateBuildPlan(prompt, images) {
  const content = [{ type: "text", text: `Build request: ${prompt}` }];
  for (const img of images) {
    if (img) {
      content.push({ type: "image_url", image_url: { url: `data:${img.mimeType};base64,${img.base64}` } });
    }
  }

  const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${GROQ_API_KEY.trim()}`,
    },
    body: JSON.stringify({
      model: GROQ_VISION_MODEL,
      max_tokens: 980, // this model's free tier caps at 1000 output tokens/minute
      reasoning_effort: "none", // skip "thinking" tokens — we need the budget for the JSON answer itself
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content },
      ],
    }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Groq API error ${response.status}: ${errText.slice(0, 300)}`);
  }

  const data = await response.json();
  const text = data.choices?.[0]?.message?.content;
  if (!text) throw new Error("No text in Groq response");
  return extractJson(text);
}

const ALLOWED_CLASSES = new Set(["Part", "WedgePart", "CornerWedgePart", "TrussPart", "Seat", "VehicleSeat"]);
const ALLOWED_SHAPES = new Set(["Block", "Ball", "Cylinder"]);
const ALLOWED_MESH_TYPES = new Set([
  "Wedge", "CornerWedge", "Prism", "Pyramid", "ParallelRamp", "RightAngleRamp", "Torso", "Head", "Brick",
]);
const ALLOWED_MATERIALS = new Set(["Plastic", "Wood", "Brick", "Concrete", "Metal", "Glass", "Grass", "Fabric"]);

function clamp(n, min, max) {
  n = Number(n);
  if (Number.isNaN(n)) return min;
  return Math.min(Math.max(n, min), max);
}

function sanitizePlan(plan) {
  if (!plan || !Array.isArray(plan.objects)) {
    return { objects: [], notes: "" };
  }

  const objects = plan.objects.slice(0, 40).flatMap((obj) => {
    if (!obj || !ALLOWED_CLASSES.has(obj.class)) return [];
    if (!Array.isArray(obj.size) || obj.size.length !== 3) return [];
    if (!Array.isArray(obj.position) || obj.position.length !== 3) return [];

    return [{
      class: obj.class,
      shape: ALLOWED_SHAPES.has(obj.shape) ? obj.shape : "Block",
      meshType: ALLOWED_MESH_TYPES.has(obj.meshType) ? obj.meshType : null,
      size: obj.size.map((v) => clamp(v, 0.3, 40)),
      position: [
        clamp(obj.position[0], -60, 60),
        clamp(obj.position[1], 0, 80),
        clamp(obj.position[2], -60, 60),
      ],
      color: Array.isArray(obj.color) && obj.color.length === 3
        ? obj.color.map((v) => clamp(v, 0, 255))
        : [155, 155, 155],
      material: ALLOWED_MATERIALS.has(obj.material) ? obj.material : "Plastic",
      name: typeof obj.name === "string" ? obj.name.slice(0, 32) : "AIPart",
      light: (obj.light && (obj.light.type === "Point" || obj.light.type === "Spot"))
        ? {
            type: obj.light.type,
            color: Array.isArray(obj.light.color) && obj.light.color.length === 3
              ? obj.light.color.map((v) => clamp(v, 0, 255))
              : [255, 240, 200],
            brightness: clamp(obj.light.brightness, 1, 5),
          }
        : null,
    }];
  });

  const notes = typeof plan.notes === "string" ? plan.notes.slice(0, 200) : "";
  return { objects, notes };
}

app.post("/build", async (req, res) => {
  if (isRateLimited()) {
    return res.status(429).json({ objects: [], notes: "Please wait about a minute between builds (free API rate limit)." });
  }

  const prompt = typeof req.body.prompt === "string" ? req.body.prompt.slice(0, 200) : "";
  if (!prompt) {
    return res.status(400).json({ objects: [], notes: "Missing prompt." });
  }

  try {
    const searchQuery = simplifyForImageSearch(prompt);
    const imageUrls = await searchImages(searchQuery);
    const images = await Promise.all(imageUrls.map(fetchImageAsBase64));
    const rawPlan = await generateBuildPlan(prompt, images.filter(Boolean));
    const safePlan = sanitizePlan(rawPlan);
    console.log(`Build "${prompt}" -> ${safePlan.objects.length} objects (search: "${searchQuery}", ${images.filter(Boolean).length} photos)`);
    return res.json(safePlan);
  } catch (err) {
    console.error("Build failed:", err.message);
    return res.status(500).json({ objects: [], notes: "Build generation failed." });
  }
});

app.listen(PORT, () => {
  console.log(`AI Builder backend listening on port ${PORT}`);
});

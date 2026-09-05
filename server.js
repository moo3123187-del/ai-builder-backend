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

const requestLog = new Map();
const MIN_INTERVAL_MS = 8_000;
function isRateLimited(ip) {
  const last = requestLog.get(ip);
  const now = Date.now();
  if (last && now - last < MIN_INTERVAL_MS) return true;
  requestLog.set(ip, now);
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
      `https://api.pexels.com/v1/search?query=${encodeURIComponent(query)}&per_page=3`,
      { headers: { Authorization: PEXELS_API_KEY.trim() } }
    );
    if (!res.ok) return [];
    const data = await res.json();
    return (data.photos || []).map((p) => p.src.medium);
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

const SYSTEM_PROMPT = `You are a 3D building planner for Roblox. You will be shown reference photos of a real object and must convert it into a JSON list of parts approximating its structure, proportions, and colors.

Available classes:
- "Part" — general block/ball/cylinder geometry, or with a "meshType" for extra shapes: "Wedge", "CornerWedge", "Prism", "Pyramid", "ParallelRamp", "RightAngleRamp", "Torso", "Head", "Brick"
- "WedgePart" — sloped ramp
- "CornerWedgePart" — corner-cut piece
- "TrussPart" — scaffolding/lattice beam
- "Seat" — a single sittable seat
- "VehicleSeat" — a driver's seat

Respond with ONLY a single valid JSON object, nothing else. Schema:
{
  "objects": [
    {
      "class": "Part" | "WedgePart" | "CornerWedgePart" | "TrussPart" | "Seat" | "VehicleSeat",
      "shape": "Block" | "Ball" | "Cylinder",
      "meshType": "Wedge" | "CornerWedge" | "Prism" | "Pyramid" | "ParallelRamp" | "RightAngleRamp" | "Torso" | "Head" | "Brick" | null,
      "size": [x, y, z],
      "position": [x, y, z],
      "color": [r, g, b],
      "material": "Plastic" | "Wood" | "Brick" | "Concrete" | "Metal" | "Glass" | "Grass" | "Fabric",
      "name": "short label, max 32 chars"
    }
  ],
  "notes": "one short sentence describing what you built and what the photos showed"
}

Rules:
- Base shapes/proportions on what's actually visible in the reference photos.
- Use 8-18 objects for anything more complex than a single simple item. Build a genuinely complete, recognizable structure -- all major walls/sections/features -- never just one block. Stay concise: this has a strict output budget, so do not exceed 18 objects and keep names short.
- Use the variety of classes and mesh types available wherever they make the result more accurate, not just plain Part blocks for everything.
- Keep content appropriate for a general, all-ages audience.
- Your entire reply must be exactly one JSON object and nothing else -- no prose, no markdown fences.`;

function extractJson(text) {
  const cleaned = text.replace(/```json/gi, "").replace(/```/g, "").trim();
  try {
    return JSON.parse(cleaned);
  } catch (e) {
    const start = cleaned.indexOf("{");
    const end = cleaned.lastIndexOf("}");
    if (start === -1 || end === -1 || end <= start) {
      throw new Error("No JSON object found in model response");
    }
    return JSON.parse(cleaned.slice(start, end + 1));
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
      max_tokens: 900, // this model's free tier caps at 1000 output tokens/minute
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
    }];
  });

  const notes = typeof plan.notes === "string" ? plan.notes.slice(0, 200) : "";
  return { objects, notes };
}

app.post("/build", async (req, res) => {
  const ip = req.ip;
  if (isRateLimited(ip)) {
    return res.status(429).json({ objects: [], notes: "Rate limited." });
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

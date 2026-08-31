/**
 * server.js
 *
 * Free-tier backend: searches Pexels for real reference photos of the
 * prompt, sends them + the prompt to Groq (vision-capable model), and
 * asks it to return a strict JSON build plan.
 *
 * Setup:
 *   npm install
 *   Set env vars: PEXELS_API_KEY, GROQ_API_KEY
 *   npm start
 */

require("dotenv").config();
const express = require("express");
const fetch = require("node-fetch");

const app = express();
app.use(express.json({ limit: "10kb" }));

const PEXELS_API_KEY = process.env.PEXELS_API_KEY;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
// Groq's vision-capable model lineup changes fairly often — llama-4-scout
// was deprecated June 17 2026. If this one stops working, check
// https://console.groq.com/docs/deprecations and
// https://console.groq.com/docs/vision for the current replacement.
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

// ------------------------------------------------------------
// Step 1: find a couple of real reference photos (Pexels, free)
// Strip size/color filler words from the SEARCH query only (not the
// prompt sent to the AI) — "a small red castle" searches better as
// just "castle".
// ------------------------------------------------------------
const FILLER_WORDS = [
  "a", "an", "the", "small", "big", "large", "tiny", "huge", "giant",
  "mini", "little", "enormous", "massive", "cozy", "compact",
  "red", "blue", "green", "yellow", "orange", "purple", "pink",
  "black", "white", "brown", "gray", "grey", "silver", "gold", "golden",
];

function simplifyForImageSearch(prompt) {
  const words = prompt
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 0 && !FILLER_WORDS.includes(w));
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

// ------------------------------------------------------------
// Step 2: ask Groq to look at the photos and produce a build plan
// ------------------------------------------------------------
const SYSTEM_PROMPT = `You are a 3D building planner for Roblox. You will be shown reference photos of a real object and asked to convert it into a JSON list of simple block/cylinder/ball shapes that approximate its structure, proportions, and colors.

Respond with ONLY valid JSON, no prose, no markdown fences. Schema:
{
  "objects": [
    {
      "class": "Part" | "WedgePart",
      "shape": "Block" | "Ball" | "Cylinder",
      "size": [x, y, z],       // studs (1 stud ≈ 1 foot). Between 0.3 and 40 each.
      "position": [x, y, z],   // studs. Center the whole object near x=0,z=0. y=0 is ground level; keep everything y >= 0.
      "color": [r, g, b],      // 0-255, based on what you actually see in the photos
      "material": "Plastic" | "Wood" | "Brick" | "Concrete" | "Metal" | "Glass" | "Grass" | "Fabric",
      "name": "short label, max 32 chars"
    }
  ],
  "notes": "one short sentence describing what you built and what the photos showed"
}

Rules:
- Base the shapes and proportions on what's actually visible in the reference photos, not a generic guess.
- Use 15-40 objects for anything more complex than a single simple item — aim for a genuinely complete, recognizable structure (all major walls/sections/features), not just a rough block.
- Keep all content appropriate for a general, all-ages audience.
- If no useful photos were provided, do your best from the text description alone.
- Output the complete JSON object. Do not truncate or stop partway through the objects array.`;

async function generateBuildPlan(prompt, images) {
  const content = [{ type: "text", text: `Build request: ${prompt}` }];
  for (const img of images) {
    if (img) {
      content.push({
        type: "image_url",
        image_url: { url: `data:${img.mimeType};base64,${img.base64}` },
      });
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
      response_format: { type: "json_object" },
      max_tokens: 4096, // headroom for up to 40 objects without truncation
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

  return JSON.parse(text);
}

// ------------------------------------------------------------
// Step 3: re-validate before returning to Roblox
// ------------------------------------------------------------
const ALLOWED_CLASSES = new Set(["Part", "WedgePart"]);
const ALLOWED_SHAPES = new Set(["Block", "Ball", "Cylinder"]);
const ALLOWED_MATERIALS = new Set([
  "Plastic", "Wood", "Brick", "Concrete", "Metal", "Glass", "Grass", "Fabric",
]);

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

    return [
      {
        class: obj.class,
        shape: ALLOWED_SHAPES.has(obj.shape) ? obj.shape : "Block",
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
      },
    ];
  });

  const notes = typeof plan.notes === "string" ? plan.notes.slice(0, 200) : "";
  return { objects, notes };
}

// ------------------------------------------------------------
// Route
// ------------------------------------------------------------
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
  console.log(`AI Builder backend (free tier) listening on port ${PORT}`);
});

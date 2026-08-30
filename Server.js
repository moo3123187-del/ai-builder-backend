/**
 * server.js
 *
 * Free-tier backend: searches Pexels for real reference photos of the
 * prompt, sends them + the prompt to Groq (vision-capable model), and
 * asks it to return a strict JSON build plan. Roblox calls this over
 * HTTPS and builds real Parts from the result.
 *
 * Cost: $0 on both APIs at normal hobby-project volume.
 *   - Pexels: free API key, ~200 requests/hour
 *   - Groq: free tier, ~30 requests/min, no credit card, no phone verification
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
// Groq's vision-capable model lineup shifts fairly often — if this
// model ID stops working, check https://console.groq.com/docs/vision
// for the current one and swap it in here.
const GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct";
const PORT = process.env.PORT || 3000;

// Simple per-IP rate limit so nobody can hammer your free quota dry.
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
// ------------------------------------------------------------
async function searchImages(query) {
  if (!PEXELS_API_KEY) return [];
  try {
    const res = await fetch(
      `https://api.pexels.com/v1/search?query=${encodeURIComponent(query)}&per_page=3`,
      { headers: { Authorization: PEXELS_API_KEY } }
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
- Use at most 40 objects. Prefer a recognizable silhouette over excessive detail.
- Keep all content appropriate for a general, all-ages audience.
- If no useful photos were provided, do your best from the text description alone.`;

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
      Authorization: `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model: GROQ_VISION_MODEL,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content },
      ],
    }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Groq API error ${response.status}: ${errText.slice(0, 200)}`);
  }

  const data = await response.json();
  const text = data.choices?.[0]?.message?.content;
  if (!text) throw new Error("No text in Groq response");

  return JSON.parse(text);
}

// ------------------------------------------------------------
// Step 3: re-validate before returning to Roblox (defense in depth —
// Roblox re-validates again too).
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
    const imageUrls = await searchImages(prompt);
    const images = await Promise.all(imageUrls.map(fetchImageAsBase64));
    const rawPlan = await generateBuildPlan(prompt, images.filter(Boolean));
    const safePlan = sanitizePlan(rawPlan);
    return res.json(safePlan);
  } catch (err) {
    console.error("Build failed:", err.message);
    return res.status(500).json({ objects: [], notes: "Build generation failed." });
  }
});

app.listen(PORT, () => {
  console.log(`AI Builder backend (free tier) listening on port ${PORT}`);
});

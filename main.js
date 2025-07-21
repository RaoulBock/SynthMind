let model;
let phrases = [];
let responses = [];

// 🧠 Load memory
function loadMemory() {
  phrases = JSON.parse(localStorage.getItem("phrases") || "[]");
  responses = JSON.parse(localStorage.getItem("responses") || "[]");
  updateMemoryExplorer();
}

// 💾 Save memory
function saveMemory() {
  localStorage.setItem("phrases", JSON.stringify(phrases));
  localStorage.setItem("responses", JSON.stringify(responses));
}

// 📜 Update memory explorer UI
function updateMemoryExplorer() {
  const list = document.getElementById("memory-list");
  list.innerHTML = "";
  phrases.forEach((phrase, i) => {
    const item = document.createElement("li");
    item.textContent = `"${phrase}" → ${responses[i]}`;
    list.appendChild(item);
  });
}

// 🔢 Encode text
function encodeText(text) {
  const tokens = text.toLowerCase().split(/\s+/);
  const value = tokens.length % 10;
  return tf.tensor2d([[value]]);
}

// 🔁 One-hot encoder
function toOneHot(index, length) {
  const arr = Array(length).fill(0);
  arr[index] = 1;
  return arr;
}

// 🧠 Train the brain
async function trainModel() {
  if (phrases.length < 2 || responses.length < 2) return;

  const xs = tf.tensor2d(
    phrases.map((_, i) => [i]),
    [phrases.length, 1]
  );
  const ysArray = responses.map((_, i) => toOneHot(i, responses.length));
  const ys = tf.tensor2d(ysArray, [responses.length, responses.length]);

  model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 10, inputShape: [1], activation: "relu" })
  );
  model.add(
    tf.layers.dense({ units: responses.length, activation: "softmax" })
  );
  model.compile({ optimizer: "adam", loss: "categoricalCrossentropy" });

  await model.fit(xs, ys, { epochs: 100 });
}

// 🚀 Handle user input
async function handleInput() {
  const inputText = document.getElementById("input").value.trim();
  if (!inputText) return;

  let response;
  if (!phrases.includes(inputText)) {
    phrases.push(inputText);
    response = `You said: "${inputText}"`;
    responses.push(response);
    saveMemory();
    updateMemoryExplorer();
    await trainModel();
  }

  if (!model || typeof model.predict !== "function" || responses.length < 2) {
    document.getElementById("response").innerText =
      "Still warming up... say more to teach me!";
    return;
  }

  const encoded = encodeText(inputText);
  const prediction = model.predict(encoded);
  const predictionArray = await prediction.array();
  const idx = predictionArray[0].indexOf(Math.max(...predictionArray[0]));
  response = responses[idx];

  document.getElementById("response").innerText = response;

  const chatLog = JSON.parse(localStorage.getItem("chatLog") || "[]");
  chatLog.push({ prompt: inputText, response });
  localStorage.setItem("chatLog", JSON.stringify(chatLog));

  Plotly.newPlot(
    "visualization",
    [
      {
        x: responses,
        y: predictionArray[0],
        type: "bar",
      },
    ],
    {
      title: "SynthMind's Thought Confidence",
    }
  );
}

// 🌀 Autonomous Thought Loop
setInterval(() => {
  if (!model || phrases.length < 2) return;

  const thought = phrases[Math.floor(Math.random() * phrases.length)];
  simulateSelfThinking(thought);
}, 15000); // every 15 seconds

// 🤖 Internal thought simulation
async function simulateSelfThinking(thought) {
  const encoded = encodeText(thought);
  const prediction = model.predict(encoded);
  const predictionArray = await prediction.array();
  const idx = predictionArray[0].indexOf(Math.max(...predictionArray[0]));
  const response = responses[idx];

  const chatLog = JSON.parse(localStorage.getItem("chatLog") || "[]");
  chatLog.push({ prompt: thought, response });
  localStorage.setItem("chatLog", JSON.stringify(chatLog));

  document.getElementById("response").innerText = `🧠 Thought: ${response}`;
  Plotly.newPlot(
    "visualization",
    [
      {
        x: responses,
        y: predictionArray[0],
        type: "bar",
      },
    ],
    {
      title: "SynthMind's Internal Thought Process",
    }
  );
}

// 🧠 Initialize
loadMemory();
trainModel();

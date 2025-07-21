let model;
let phrases = [];
let responses = [];

function loadMemory() {
  phrases = JSON.parse(localStorage.getItem("phrases") || "[]");
  responses = JSON.parse(localStorage.getItem("responses") || "[]");
  updateMemoryExplorer();
}

function saveMemory() {
  localStorage.setItem("phrases", JSON.stringify(phrases));
  localStorage.setItem("responses", JSON.stringify(responses));
}

function updateMemoryExplorer() {
  const list = document.getElementById("memory-list");
  list.innerHTML = "";
  phrases.forEach((phrase, i) => {
    const item = document.createElement("li");
    item.textContent = `"${phrase}" → ${responses[i]}`;
    list.appendChild(item);
  });
}

function trimMemoryIfNeeded() {
  const maxLength = 500;
  if (phrases.length > maxLength) {
    phrases = phrases.slice(-maxLength);
    responses = responses.slice(-maxLength);
    saveMemory();
    updateMemoryExplorer();
  }
}

function encodeText(text) {
  const tokens = text.toLowerCase().split(/\s+/);
  const value = tokens.length % 10;
  return tf.tensor2d([[value]]);
}

function toOneHot(index, length) {
  const arr = Array(length).fill(0);
  arr[index] = 1;
  return arr;
}

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

function craftResponse(base, input) {
  if (input.toLowerCase().includes("sad")) {
    return `💬 "${input}" — I’m here for you. Sadness is a signal, not a flaw.`;
  } else if (
    input.toLowerCase().includes("joke") ||
    input.toLowerCase().includes("funny")
  ) {
    return `🃏 "${input}" — Why did the neural net cross the road? It had data on the other side!`;
  } else if (input.toLowerCase().includes("fact")) {
    return `📚 "${input}" — Honey never spoils. Nature’s original preservative.`;
  }
  return `🧠 "${input}" — ${base}`;
}

async function handleInput() {
  const inputText = document.getElementById("input").value.trim();
  if (!inputText) return;

  let baseResponse;
  if (!phrases.includes(inputText)) {
    phrases.push(inputText);
    baseResponse = `You said: "${inputText}"`;
    responses.push(baseResponse);
    saveMemory();
    trimMemoryIfNeeded();
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
  baseResponse = responses[idx];

  const finalResponse = craftResponse(baseResponse, inputText);
  document.getElementById("response").innerText = finalResponse;

  const chatLog = JSON.parse(localStorage.getItem("chatLog") || "[]");
  chatLog.push({ prompt: inputText, response: finalResponse });
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
      title: "SynthMind Confidence Chart",
    }
  );
}

setInterval(() => {
  if (!model || phrases.length < 2) return;
  const thought = phrases[Math.floor(Math.random() * phrases.length)];
  simulateSelfThinking(thought);
}, 15000);

async function simulateSelfThinking(thought) {
  const encoded = encodeText(thought);
  const prediction = model.predict(encoded);
  const predictionArray = await prediction.array();
  const idx = predictionArray[0].indexOf(Math.max(...predictionArray[0]));
  const baseResponse = responses[idx];
  const finalResponse = craftResponse(baseResponse, thought);

  phrases.push(thought);
  responses.push(baseResponse);
  saveMemory();
  trimMemoryIfNeeded();
  updateMemoryExplorer();

  const chatLog = JSON.parse(localStorage.getItem("chatLog") || "[]");
  chatLog.push({ prompt: thought, response: finalResponse });
  localStorage.setItem("chatLog", JSON.stringify(chatLog));

  document.getElementById(
    "response"
  ).innerText = `🧠 Thought: ${finalResponse}`;
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
      title: "SynthMind Internal Thought",
    }
  );
}

loadMemory();
trainModel();

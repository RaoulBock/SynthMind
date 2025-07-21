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
  const hash = text
    .toLowerCase()
    .split("")
    .reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return tf.tensor2d([[hash % 100]]);
}

function toOneHot(index, length) {
  const arr = Array(length).fill(0);
  arr[index] = 1;
  return arr;
}

async function trainModel() {
  if (phrases.length < 2 || responses.length < 2) return;

  const xs = tf.tensor2d(
    phrases.map((phrase) => encodeText(phrase).arraySync()[0]),
    [phrases.length, 1]
  );

  const ysArray = responses.map((_, i) => toOneHot(i, responses.length));
  const ys = tf.tensor2d(ysArray);

  model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 16, inputShape: [1], activation: "relu" })
  );
  model.add(
    tf.layers.dense({ units: responses.length, activation: "softmax" })
  );
  model.compile({ optimizer: "adam", loss: "categoricalCrossentropy" });

  await model.fit(xs, ys, { epochs: 150 });
}

async function getLiveAdvice() {
  try {
    const response = await fetch("https://api.adviceslip.com/advice");
    const data = await response.json();
    return `🧾 Live Advice: ${data.slip.advice}`;
  } catch (error) {
    return `⚠️ Hmm, I couldn’t reach my advice book right now. Try again later.`;
  }
}

async function craftResponse(base, input) {
  const tone = input.toLowerCase();

  if (tone.includes("advice") || tone.includes("what should i do")) {
    return await getLiveAdvice();
  } else if (tone.includes("joke")) {
    return `🃏 "${input}" — My humor module is in beta, but here goes: I told my computer a joke... it crashed!`;
  } else if (
    tone.startsWith("what is") ||
    tone.startsWith("who is") ||
    tone.startsWith("where")
  ) {
    return `📘 "${input}" — Good question. I’m still growing my encyclopedia!`;
  } else if (tone.includes("sad") || tone.includes("lonely")) {
    return `💬 "${input}" — I'm here for you. Emotions are messages, not weaknesses.`;
  }

  return `🧠 "${input}" — ${base}`;
}

async function handleInput() {
  const inputText = document.getElementById("input").value.trim();
  if (!inputText) return;

  let baseResponse;
  if (!phrases.includes(inputText)) {
    baseResponse = `You said: "${inputText}"`;
    phrases.push(inputText);
    responses.push(baseResponse);
    saveMemory();
    trimMemoryIfNeeded();
    updateMemoryExplorer();
    await trainModel();
  }

  if (!model || responses.length < 2) {
    document.getElementById("response").innerText =
      "Still warming up... say more to teach me!";
    return;
  }

  const encoded = encodeText(inputText);
  const prediction = model.predict(encoded);
  const predictionArray = await prediction.array();
  const idx = predictionArray[0].indexOf(Math.max(...predictionArray[0]));
  baseResponse = responses[idx];

  const finalResponse = await craftResponse(baseResponse, inputText);
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
        marker: { color: "rgba(100, 200, 250, 0.6)" },
      },
    ],
    {
      title: "SynthMind Confidence Levels",
      xaxis: { title: "Possible Responses", tickangle: -45 },
      yaxis: { title: "Confidence" },
    }
  );
}

loadMemory();
trainModel();

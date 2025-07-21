let model;
let phrases = [];
let responses = [];

function loadMemory() {
  phrases = JSON.parse(localStorage.getItem("phrases") || "[]");
  responses = JSON.parse(localStorage.getItem("responses") || "[]");
}

function saveMemory() {
  localStorage.setItem("phrases", JSON.stringify(phrases));
  localStorage.setItem("responses", JSON.stringify(responses));
}

function trimMemoryIfNeeded() {
  const maxLength = 500;
  if (phrases.length > maxLength) {
    phrases = phrases.slice(-maxLength);
    responses = responses.slice(-maxLength);
    saveMemory();
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
  } catch {
    return `⚠️ Couldn't connect to the wisdom cloud. Try again later.`;
  }
}

async function craftResponse(base, input) {
  const tone = input.toLowerCase();

  const mathMatch = tone.match(/^(\d+)\s*[\+\-\*\/]\s*(\d+)$/);
  if (mathMatch) {
    try {
      const result = eval(tone);
      return `🧮 "${input}" — The answer is ${result}.`;
    } catch {
      return `⚠️ I couldn’t compute that expression.`;
    }
  }

  if (tone.includes("advice") || tone.includes("what should i do")) {
    return await getLiveAdvice();
  }

  if (tone.includes("joke") || tone.includes("funny")) {
    return `🃏 "${input}" — Why did the robot bring a ladder? To reach cloud storage.`;
  }

  if (
    tone.startsWith("what is") ||
    tone.startsWith("who is") ||
    tone.startsWith("where is")
  ) {
    return `📘 "${input}" — Good question. I’m still expanding my encyclopedia!`;
  }

  if (tone.includes("sad") || tone.includes("lonely")) {
    return `💬 "${input}" — I'm here for you. Emotions are signals, not errors.`;
  }

  return `🧠 "${input}" — ${base}`;
}

function appendMessage(text, sender) {
  const msg = document.createElement("div");
  msg.classList.add(
    "message",
    sender === "user" ? "user-message" : "bot-message"
  );
  msg.textContent = text;
  document.getElementById("chat-log").appendChild(msg);
  document.getElementById("chat-window").scrollTop =
    document.getElementById("chat-window").scrollHeight;
}

async function handleInput() {
  const inputText = document.getElementById("input").value.trim();
  if (!inputText) return;

  appendMessage(inputText, "user");

  let baseResponse;
  if (!phrases.includes(inputText)) {
    baseResponse = `You said: "${inputText}"`;
    phrases.push(inputText);
    responses.push(baseResponse);
    saveMemory();
    trimMemoryIfNeeded();
    await trainModel();
  }

  if (!model || responses.length < 2) {
    appendMessage("Still warming up... say more to teach me!", "bot");
    return;
  }

  const encoded = encodeText(inputText);
  const prediction = model.predict(encoded);
  const predictionArray = await prediction.array();
  const idx = predictionArray[0].indexOf(Math.max(...predictionArray[0]));
  baseResponse = responses[idx];

  const finalResponse = await craftResponse(baseResponse, inputText);
  appendMessage(finalResponse, "bot");

  const chatLog = JSON.parse(localStorage.getItem("chatLog") || "[]");
  chatLog.push({ prompt: inputText, response: finalResponse });
  localStorage.setItem("chatLog", JSON.stringify(chatLog));
}

loadMemory();
trainModel();

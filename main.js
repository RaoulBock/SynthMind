let model;
let phrases = [];
let responses = [];

// 🧠 Load memory from localStorage
function loadMemory() {
  const storedPhrases = localStorage.getItem("phrases");
  const storedResponses = localStorage.getItem("responses");

  if (storedPhrases && storedResponses) {
    phrases = JSON.parse(storedPhrases);
    responses = JSON.parse(storedResponses);
  }
}

// 💾 Save memory to localStorage
function saveMemory() {
  localStorage.setItem("phrases", JSON.stringify(phrases));
  localStorage.setItem("responses", JSON.stringify(responses));
}

// 🔢 Encode text (simple token feature)
function encodeText(text) {
  const tokens = text.toLowerCase().split(/\s+/);
  const value = tokens.length % 10;
  return tf.tensor2d([[value]]);
}

// 🔁 One-hot encode target labels
function toOneHot(index, length) {
  const arr = Array(length).fill(0);
  arr[index] = 1;
  return arr;
}

// 🧠 Train the neural model
async function trainModel() {
  const numSamples = phrases.length;
  const numClasses = responses.length;

  if (numSamples < 2 || numClasses < 2) return;

  const xs = tf.tensor2d(
    phrases.map((_, i) => [i]),
    [numSamples, 1]
  );
  const ys = tf.tensor2d(
    responses.map((_, i) => toOneHot(i, numClasses)),
    [numSamples, numClasses]
  );

  model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 10, inputShape: [1], activation: "relu" })
  );
  model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));
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
    saveMemory(); // 👉 Save every new interaction
    await trainModel();
  }

  if (!model || typeof model.predict !== "function" || responses.length < 2) {
    document.getElementById("response").innerText =
      "Still warming up... say a bit more to teach me!";
    return;
  }

  const encoded = encodeText(inputText);
  const prediction = model.predict(encoded);
  const predictionArray = await prediction.array();
  const idx = predictionArray[0].indexOf(Math.max(...predictionArray[0]));
  response = responses[idx];

  document.getElementById("response").innerText = response;

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
      title: "Brain Activity – Confidence in Responses",
    }
  );
}

// 🧠 Bootstrap memory and brain
loadMemory();
trainModel();

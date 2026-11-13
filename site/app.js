const MODEL_SIZE = 640;
const CONFIDENCE = 0.30;
const IOU_LIMIT = 0.45;
const CLASSES = [
  "barricade", "dumpster", "excavators", "gloves", "hardhat", "mask",
  "no-hardhat", "no-mask", "no-safety vest", "person", "safety net",
  "safety shoes", "safety vest", "dump truck", "mini-van", "truck", "wheel loader"
];
const ALERT_CLASSES = new Set(["no-hardhat", "no-mask", "no-safety vest"]);
const VIDEO_SAMPLE = "assets/site_clip.mp4";
const RUNTIME = new URL("vendor/", document.baseURI).href;

const $ = (selector) => document.querySelector(selector);
const resultCanvas = $("#resultCanvas");
const resultContext = resultCanvas.getContext("2d");
const modelCanvas = document.createElement("canvas");
modelCanvas.width = MODEL_SIZE;
modelCanvas.height = MODEL_SIZE;
const modelContext = modelCanvas.getContext("2d", { willReadFrequently: true });

let session;
let sourceVideo;
let sourceObjectUrl;
let mode = "idle";
let runId = 0;
let counts = {};
let concerns = {};
let framesAnalyzed = 0;
let frameHandler;
let recorder;

function setStatus(message) {
  $("#status").textContent = message;
}

function showLoading(show) {
  $("#loading").classList.toggle("active", show);
  $("#modelUnit").classList.toggle("running", show);
  $("#modelUnit").classList.toggle("complete", !show && !!session);
  $("#modelPhase").textContent = show ? "Finding patterns…" : "Ready for a scene";
}

async function loadModel() {
  if (!window.ort) throw new Error("ONNX Runtime could not load. Check your connection.");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = RUNTIME;
  setStatus("Loading the trained modelâ€¦");
  session = await ort.InferenceSession.create("models/best.onnx", {
    executionProviders: ["wasm"]
  });
  setStatus("Model ready. Choose an image or video, or try a sample.");
  $("#modelPhase").textContent = "Ready for a scene";
}

function prepareInput(image) {
  const width = image.videoWidth || image.naturalWidth || image.width;
  const height = image.videoHeight || image.naturalHeight || image.height;
  const scale = Math.min(MODEL_SIZE / width, MODEL_SIZE / height);
  const drawnWidth = Math.round(width * scale);
  const drawnHeight = Math.round(height * scale);
  const padX = Math.round((MODEL_SIZE - drawnWidth) / 2);
  const padY = Math.round((MODEL_SIZE - drawnHeight) / 2);

  modelContext.fillStyle = "#727272";
  modelContext.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  modelContext.drawImage(image, padX, padY, drawnWidth, drawnHeight);

  const rgba = modelContext.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const pixels = MODEL_SIZE * MODEL_SIZE;
  const input = new Float32Array(3 * pixels);
  for (let pixel = 0; pixel < pixels; pixel++) {
    input[pixel] = rgba[pixel * 4] / 255;
    input[pixels + pixel] = rgba[pixel * 4 + 1] / 255;
    input[2 * pixels + pixel] = rgba[pixel * 4 + 2] / 255;
  }
  return { input, width, height, scale, padX, padY };
}

function intersectionOverUnion(a, b) {
  const left = Math.max(a.x1, b.x1);
  const top = Math.max(a.y1, b.y1);
  const right = Math.min(a.x2, b.x2);
  const bottom = Math.min(a.y2, b.y2);
  const overlap = Math.max(0, right - left) * Math.max(0, bottom - top);
  const aArea = (a.x2 - a.x1) * (a.y2 - a.y1);
  const bArea = (b.x2 - b.x1) * (b.y2 - b.y1);
  return overlap / (aArea + bArea - overlap || 1);
}

function decodeOutput(tensor, source) {
  const dimensions = tensor.dims;
  const channelsFirst = dimensions[1] === CLASSES.length + 4;
  const candidates = channelsFirst ? dimensions[2] : dimensions[1];
  const channels = CLASSES.length + 4;
  if (!channelsFirst && dimensions[2] !== channels) {
    throw new Error(`Unexpected model output: ${dimensions.join(" Ã— ")}`);
  }
  const value = (channel, candidate) => channelsFirst
    ? tensor.data[channel * candidates + candidate]
    : tensor.data[candidate * channels + channel];

  const boxes = [];
  for (let index = 0; index < candidates; index++) {
    let classId = 0;
    let score = 0;
    for (let classIndex = 0; classIndex < CLASSES.length; classIndex++) {
      const probability = value(classIndex + 4, index);
      if (probability > score) {
        score = probability;
        classId = classIndex;
      }
    }
    if (score < CONFIDENCE) continue;

    const cx = value(0, index);
    const cy = value(1, index);
    const width = value(2, index);
    const height = value(3, index);
    const x1 = Math.max(0, Math.min(source.width, (cx - width / 2 - source.padX) / source.scale));
    const y1 = Math.max(0, Math.min(source.height, (cy - height / 2 - source.padY) / source.scale));
    const x2 = Math.max(0, Math.min(source.width, (cx + width / 2 - source.padX) / source.scale));
    const y2 = Math.max(0, Math.min(source.height, (cy + height / 2 - source.padY) / source.scale));
    if (x2 <= x1 || y2 <= y1) continue;
    boxes.push({ classId, label: CLASSES[classId], score, x1, y1, x2, y2 });
  }

  boxes.sort((a, b) => b.score - a.score);
  const selected = [];
  for (const box of boxes.slice(0, 300)) {
    if (selected.every((kept) => kept.classId !== box.classId || intersectionOverUnion(box, kept) < IOU_LIMIT)) {
      selected.push(box);
    }
    if (selected.length >= 100) break;
  }
  return selected;
}

async function detect(image) {
  const source = prepareInput(image);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const tensor = new ort.Tensor("float32", source.input, [1, 3, MODEL_SIZE, MODEL_SIZE]);
  const output = await session.run({ [inputName]: tensor });
  return decodeOutput(output[outputName], source);
}

function drawResult(image, boxes) {
  const width = image.videoWidth || image.naturalWidth || image.width;
  const height = image.videoHeight || image.naturalHeight || image.height;
  resultCanvas.width = width;
  resultCanvas.height = height;
  resultCanvas.hidden = false;
  $("#empty").hidden = true;
  resultContext.drawImage(image, 0, 0, width, height);

  const line = Math.max(2, Math.round(Math.min(width, height) / 240));
  resultContext.lineWidth = line;
  resultContext.font = `bold ${Math.max(13, Math.round(width / 45))}px Arial`;
  for (const box of boxes) {
    const color = ALERT_CLASSES.has(box.label) ? "#ffb76b" : "#c4ef65";
    const label = `${box.label} ${Math.round(box.score * 100)}%`;
    const textWidth = resultContext.measureText(label).width + 13;
    const labelHeight = Math.max(21, Math.round(width / 32));
    const labelY = Math.max(0, box.y1 - labelHeight);
    resultContext.strokeStyle = color;
    resultContext.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
    resultContext.fillStyle = color;
    resultContext.fillRect(box.x1, labelY, textWidth, labelHeight);
    resultContext.fillStyle = "#172219";
    resultContext.fillText(label, box.x1 + 6, labelY + labelHeight - 6);
  }
}

function updateReport(boxes) {
  for (const box of boxes) {
    counts[box.label] = (counts[box.label] || 0) + 1;
    if (ALERT_CLASSES.has(box.label)) {
      concerns[box.label] = (concerns[box.label] || 0) + 1;
    }
  }
  $("#objectCount").textContent = Object.values(counts).reduce((sum, value) => sum + value, 0);
  $("#concernCount").textContent = Object.values(concerns).reduce((sum, value) => sum + value, 0);
  const tags = $("#tags");
  tags.replaceChildren();
  for (const [label, total] of Object.entries(counts)) {
    const tag = document.createElement("span");
    tag.className = `tag ${concerns[label] ? "alert" : ""}`;
    tag.textContent = `${label} Ã— ${total}`;
    tags.append(tag);
  }
  if (!tags.children.length) {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = "No objects detected at this confidence level";
    tags.append(tag);
  }
  $("#report").classList.add("show");
}

function showSource(media) {
  const preview = $("#sourcePreview");
  preview.replaceChildren(media);
  preview.hidden = false;
  $("#drop").classList.add("has-media");
  preview.classList.remove("arriving");
  void preview.offsetWidth;
  preview.classList.add("arriving");
}

function reset() {
  runId++;
  if (recorder && recorder.state !== "inactive") recorder.stop();
  recorder = null;
  if (sourceVideo) sourceVideo.pause();
  if (sourceObjectUrl) URL.revokeObjectURL(sourceObjectUrl);
  sourceVideo = null;
  frameHandler = null;
  sourceObjectUrl = null;
  counts = {};
  concerns = {};
  framesAnalyzed = 0;
  mode = "idle";
  $("#modelUnit").classList.remove("running", "complete");
  $("#report").classList.remove("show");
  $("#downloadMedia").hidden = true;
  $("#downloadReport").hidden = true;
  $("#pauseVideo").hidden = true;
}

function offerReport(type) {
  const report = {
    type,
    model: "YOLO11n construction safety detector",
    confidence_threshold: CONFIDENCE,
    analyzed_frames: framesAnalyzed,
    class_counts: counts,
    possible_ppe_concerns: concerns,
    note: "Possible concerns need human review. Video counts are detections across analyzed frames, not unique people."
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const link = $("#downloadReport");
  link.href = URL.createObjectURL(blob);
  link.download = "sitesight-report.json";
  link.hidden = false;
}

async function analyzeImage(url) {
  if (!session) { setStatus("The model is still loading. Please try again shortly."); return; }
  reset();
  const currentRun = runId;
  showLoading(true);
  setStatus("Scanning the imageâ€¦");
  const image = new Image();
  image.onload = async () => {
    if (currentRun !== runId) return;
    try {
      showSource(image);
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      const started = performance.now();
      const boxes = await detect(image);
      if (!matchMedia("(prefers-reduced-motion: reduce)").matches) await new Promise(resolve => setTimeout(resolve, Math.max(0, 1100 - (performance.now() - started))));
      if (currentRun !== runId) return;
      drawResult(image, boxes);
      framesAnalyzed = 1;
      updateReport(boxes);
      const link = $("#downloadMedia");
      link.href = resultCanvas.toDataURL("image/png");
      link.download = "sitesight-annotated.png";
      link.hidden = false;
      offerReport("image");
      setStatus(`Analysis complete Â· ${boxes.length} objects detected.`);
    } catch (error) {
      setStatus(`Analysis failed: ${error.message}`);
    } finally {
      showLoading(false);
    }
  };
  image.onerror = () => { showLoading(false); setStatus("Could not open this image."); };
  image.src = url;
}

function waitForVideo(video) {
  return new Promise((resolve, reject) => {
    video.onloadedmetadata = resolve;
    video.onerror = () => reject(new Error("Could not open this video."));
  });
}

async function analyzeVideo(url) {
  if (!session) { setStatus("The model is still loading. Please try again shortly."); return; }
  reset();
  const currentRun = runId;
  showLoading(true);
  setStatus("Loading videoâ€¦");
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.controls = false;
  sourceVideo = video;
  video.src = url;
  try {
    await waitForVideo(video);
    if (currentRun !== runId) return;
    const still = document.createElement("canvas");
    still.width = video.videoWidth;
    still.height = video.videoHeight;
    const stillContext = still.getContext("2d");
    showSource(video);
    resultCanvas.width = video.videoWidth;
    resultCanvas.height = video.videoHeight;
    $("#pauseVideo").hidden = false;
    $("#pauseVideo").textContent = "Pause video";
    mode = "video";
    showLoading(false);
    setStatus("Analyzing video frames liveâ€¦");
    await video.play();

    if (window.MediaRecorder && resultCanvas.captureStream && MediaRecorder.isTypeSupported("video/webm")) {
      const stream = resultCanvas.captureStream(15);
      const chunks = [];
      const format = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9" : "video/webm";
      recorder = new MediaRecorder(stream, { mimeType: format });
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        if (currentRun !== runId || !chunks.length) return;
        const link = $("#downloadMedia");
        link.href = URL.createObjectURL(new Blob(chunks, { type: format }));
        link.download = "sitesight-annotated.webm";
        link.textContent = "â†“ Download annotated video";
        link.hidden = false;
      };
      recorder.start(1000);
    }

    async function nextFrame() {
      if (currentRun !== runId || video.paused || video.ended) return;
      stillContext.drawImage(video, 0, 0, still.width, still.height);
      try {
        const boxes = await detect(still);
        if (currentRun !== runId) return;
        drawResult(still, boxes);
        framesAnalyzed++;
        updateReport(boxes);
        setStatus(`Analyzing video Â· ${video.currentTime.toFixed(1)} / ${video.duration.toFixed(1)} s Â· ${framesAnalyzed} frames scanned`);
      } catch (error) {
        video.pause();
        setStatus(`Video analysis stopped: ${error.message}`);
        return;
      }
      if (!video.paused && !video.ended) video.requestVideoFrameCallback(nextFrame);
    }
    frameHandler = nextFrame;
    if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(nextFrame);
    else setStatus("This browser cannot process video frames. Try a recent Chrome or Edge browser.");
    video.onended = () => {
      showLoading(false);
      $("#modelPhase").textContent = "Analysis complete";
      if (recorder && recorder.state !== "inactive") recorder.stop();
      offerReport("video");
      setStatus(`Video complete Â· ${framesAnalyzed} frames analyzed.`);
    };
  } catch (error) {
    showLoading(false);
    setStatus(error.message);
  }
}

function analyzeFile(file) {
  if (file.size > 200 * 1024 * 1024) {
    setStatus("Choose a file under 200 MB.");
    return;
  }
  if (!/\.(jpe?g|png|webp|mp4|webm|mov)$/i.test(file.name)) {
    setStatus("Choose a JPG, PNG, WebP, MP4, WebM, or MOV file.");
    return;
  }
  const url = URL.createObjectURL(file);
  if (file.type.startsWith("image/")) analyzeImage(url);
  else analyzeVideo(url);
  sourceObjectUrl = url;
}

$("#browse").addEventListener("click", (event) => { event.stopPropagation(); $("#file").click(); });
$("#drop").addEventListener("click", () => $("#file").click());
$("#drop").addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); $("#file").click(); }
});
for (const name of ["dragenter", "dragover", "dragleave", "drop"]) {
  $("#drop").addEventListener(name, (event) => {
    event.preventDefault();
    $("#drop").classList.toggle("drag", name === "dragenter" || name === "dragover");
  });
}
$("#drop").addEventListener("drop", (event) => {
  if (event.dataTransfer.files[0]) analyzeFile(event.dataTransfer.files[0]);
});
$("#file").addEventListener("change", (event) => {
  if (event.target.files[0]) analyzeFile(event.target.files[0]);
});
$("#sampleImage").addEventListener("click", () => analyzeImage("assets/sample-construction.jpg"));
$("#sampleVideo").addEventListener("click", () => analyzeVideo(VIDEO_SAMPLE));
$("#pauseVideo").addEventListener("click", () => {
  if (!sourceVideo) return;
  if (sourceVideo.paused) {
    sourceVideo.play();
    if (frameHandler && sourceVideo.requestVideoFrameCallback) sourceVideo.requestVideoFrameCallback(frameHandler);
    $("#pauseVideo").textContent = "Pause video";
  } else {
    sourceVideo.pause();
    $("#pauseVideo").textContent = "Resume video";
  }
});

window.addEventListener("load", () => loadModel().catch((error) => setStatus(error.message)));

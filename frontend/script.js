const viewfinder = document.getElementById("viewfinder");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const dropHint = document.getElementById("dropHint");
const scanline = document.getElementById("scanline");
const loading = document.getElementById("loading");
const readout = document.getElementById("readout");

const claseEs = document.getElementById("claseEs");
const claseEn = document.getElementById("claseEn");
const confVal = document.getElementById("confVal");
const heatbarFill = document.getElementById("heatbarFill");
const probList = document.getElementById("probList");
const consoleText = document.getElementById("consoleText");
const imgColor = document.getElementById("imgColor");
const imgGris = document.getElementById("imgGris");

viewfinder.addEventListener("click", () => fileInput.click());

viewfinder.addEventListener("dragover", (e) => {
  e.preventDefault();
  viewfinder.style.borderColor = "#f97316";
});
viewfinder.addEventListener("dragleave", () => {
  viewfinder.style.borderColor = "";
});
viewfinder.addEventListener("drop", (e) => {
  e.preventDefault();
  viewfinder.style.borderColor = "";
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.hidden = false;
  dropHint.hidden = true;
  scanline.hidden = false;
  readout.hidden = true;
  imgColor.src = url;

  classify(file);
}

async function classify(file) {
  loading.hidden = false;
  const formData = new FormData();
  formData.append("file", file);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000);

  try {
    const res = await fetch(`${window.API_BASE}/api/classify`, { method: "POST", body: formData, signal: controller.signal });
    if (!res.ok) throw new Error((await res.json()).detail || "Error al clasificar");
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    if (err.name === "AbortError") err = new Error("Se agotó el tiempo de espera (30s). Revisá tu conexión o la API key de Claude.");
    consoleText.textContent = `Error: ${err.message}`;
    readout.hidden = false;
  } finally {
    clearTimeout(timeoutId);
    loading.hidden = true;
    scanline.hidden = true;
  }
}

function renderResult(data) {
  claseEs.textContent = data.clase_predicha.es;
  claseEn.textContent = data.clase_predicha.en;
  const pct = Math.round(data.confianza * 100);
  confVal.textContent = `${pct}%`;
  if (data.imagen_gris_base64) {
    imgGris.src = `data:image/jpeg;base64,${data.imagen_gris_base64}`;
  }

  readout.hidden = false;
  requestAnimationFrame(() => {
    heatbarFill.style.width = `${pct}%`;
  });

  probList.innerHTML = "";
  data.probabilidades.slice(0, 6).forEach((p) => {
    const row = document.createElement("div");
    row.className = "problist__row";
    row.innerHTML = `
      <span class="problist__name">${p.es}</span>
      <div class="problist__track"><div class="problist__track-fill" style="width:${Math.round(p.prob * 100)}%"></div></div>
      <span class="problist__pct">${Math.round(p.prob * 100)}%</span>
    `;
    probList.appendChild(row);
  });

  typeExplanation(data.explicacion_claude || data.error_claude || "Sin explicación disponible.");
}

function typeExplanation(text) {
  consoleText.textContent = "";
  let i = 0;
  const interval = setInterval(() => {
    consoleText.textContent += text[i] || "";
    i++;
    if (i >= text.length) clearInterval(interval);
  }, 12);
}

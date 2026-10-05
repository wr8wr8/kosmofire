const SEVERITY_COLORS = { 1: "#ffd08a", 2: "#ff8a3d", 3: "#c1121f" };
const map = L.map("map").setView([48.5, 44.0], 6);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 18, attribution: "&copy; OpenStreetMap" }).addTo(map);

let hotspotLayer = null;
let burnLayer = null;

const form = document.getElementById("query");
const areaMode = document.getElementById("area-mode");
const bboxInput = document.getElementById("bbox");
const statusEl = document.getElementById("status");

areaMode.addEventListener("change", () => { bboxInput.hidden = areaMode.value !== "bbox"; });

function currentParams() {
  const params = new URLSearchParams();
  if (areaMode.value === "bbox" && bboxInput.value.trim()) {
    params.set("bbox", bboxInput.value.trim());
  } else {
    const b = map.getBounds();
    params.set("bbox", [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].join(","));
  }
  const from = document.getElementById("date-from").value;
  const to = document.getElementById("date-to").value;
  if (from) params.set("date_from", from);
  if (to) params.set("date_to", to);
  return params;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
  return response.json();
}

function renderReport(report) {
  const rows = report.by_severity.map((s) =>
    `<tr><td>${s.severity_name}</td><td>${s.area_ha.toLocaleString("ru-RU")}</td><td>${(s.share * 100).toFixed(1)}%</td></tr>`).join("");
  document.getElementById("report").innerHTML =
    `<table><tr><th>Степень</th><th>га</th><th>доля</th></tr>${rows}` +
    `<tr><th>Всего</th><th>${report.total_burned_area_ha.toLocaleString("ru-RU")}</th><th></th></tr></table>` +
    `<p>Термоточек: ${report.hotspot_count.toLocaleString("ru-RU")}</p>`;
}

function updateDownloads(params) {
  const q = params.toString();
  document.getElementById("dl-geojson").href = `/api/v1/burns/export?${q}&format=geojson`;
  document.getElementById("dl-shp").href = `/api/v1/burns/export?${q}&format=shp`;
  document.getElementById("dl-csv").href = `/api/v1/report?${q}&format=csv`;
  document.getElementById("dl-json").href = `/api/v1/report?${q}`;
}

async function load() {
  const params = currentParams();
  statusEl.textContent = "Загрузка…";
  try {
    const [hotspots, burns, report] = await Promise.all([
      fetchJson(`/api/v1/hotspots?${params}`),
      fetchJson(`/api/v1/burns?${params}`),
      fetchJson(`/api/v1/report?${params}`),
    ]);
    if (hotspotLayer) hotspotLayer.remove();
    if (burnLayer) burnLayer.remove();
    burnLayer = L.geoJSON(burns, {
      style: (f) => ({ color: SEVERITY_COLORS[f.properties.severity], weight: 1, fillOpacity: 0.55 }),
      onEachFeature: (f, layer) => layer.bindPopup(
        `${f.properties.contour_id}<br>Степень: ${f.properties.severity_name}<br>Площадь: ${f.properties.area_ha} га<br>Снимок: ${f.properties.date_post}`),
    }).addTo(map);
    hotspotLayer = L.geoJSON(hotspots, {
      pointToLayer: (f, latlng) => L.circleMarker(latlng, { radius: 3, color: "#ff2d00", weight: 1, fillOpacity: 0.9 }),
      onEachFeature: (f, layer) => layer.bindPopup(`Термоточка<br>${f.properties.detected_at}<br>${f.properties.satellite}`),
    }).addTo(map);
    renderReport(report);
    updateDownloads(params);
    statusEl.textContent = `Полигонов гари: ${burns.features.length}`;
  } catch (error) {
    statusEl.textContent = `Ошибка: ${error.message}`;
  }
}

form.addEventListener("submit", (event) => { event.preventDefault(); load(); });
document.getElementById("fit").addEventListener("click", async () => {
  const health = await fetchJson("/api/v1/burns");
  const layer = L.geoJSON(health);
  if (layer.getBounds().isValid()) map.fitBounds(layer.getBounds());
  load();
});
load();

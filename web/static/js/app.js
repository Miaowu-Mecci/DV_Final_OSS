/********************
 * Nyx Visualization Dashboard
 * Interactive: canvas heatmaps with tooltip + Plotly stats + preloaded 3D volumes
 * GPU: requests high-performance (discrete GPU)
 ********************/

// ===== State =====
let statsData = [];
let histogramsData = [];
let volumeData = new Array(100).fill(null);
let volumeLoaded = 0;
let volMeta = { shape: [64, 64, 64], raw_min: 0.0, raw_max: 1.0 };

let volScene, volCamera, volRenderer, volControls, volMesh;
let volRayMaterial = null, volTexture = null;
let vol3DInitialized = false;
let isPlaying = false, animTimer = null;
let currentTs = 0, currentProjType = 'mip';
let abortCtrl = null;
let projCache = new Map(); // key: "ts_axis_type" -> {matrix, min, max}
let statsInitialized = false;
let statRafId = null;
let densityFilterRaw = { min: null, max: null }; // 原始密度筛选阈值，null=无筛选

// Plotly dark theme
const PLOT_PAPER = 'rgba(0,0,0,0)';
const PLOT_BG    = 'rgba(0,0,0,0)';
const PLOT_TEXT  = '#e0e6f0';
const PLOT_GRID  = 'rgba(255,255,255,0.08)';
const PLOT_MUTED = '#8892b0';

// ===== Colormap =====
function hotColor(t) {
    t = Math.max(0, Math.min(1, t));
    let r, g, b;
    if (t < 0.33) {
        r = t / 0.33; g = 0; b = 0;
    } else if (t < 0.66) {
        r = 1; g = (t - 0.33) / 0.33; b = 0;
    } else {
        r = 1; g = 1; b = (t - 0.66) / 0.34;
    }
    return [Math.floor(r * 255), Math.floor(g * 255), Math.floor(b * 255)];
}

// ===== Ray Marching Shaders =====
const RM_VERT = `
varying vec3 vWorldPos;
void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

const RM_FRAG = `
precision highp float;
precision highp sampler3D;
varying vec3 vWorldPos;
uniform vec3 uCameraPos;
uniform vec3 uBoxMin;
uniform vec3 uBoxMax;
uniform sampler3D uVolume;
uniform float uStepSize;
uniform float uOpacityScale;
uniform float uThresholdMin;
uniform float uThresholdMax;
uniform float uBrightness;
uniform int uTFMode;
uniform float uRawMin;
uniform float uRawMax;
uniform int uLogScale;

vec2 rayBoxIntersection(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax) {
    vec3 invDir = 1.0 / rayDir;
    vec3 tMin = (boxMin - rayOrigin) * invDir;
    vec3 tMax = (boxMax - rayOrigin) * invDir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
}

vec4 tfHot(float d) {
    vec3 color; float a;
    if (d < 0.15) { color = vec3(0.0, 0.0, 0.02); a = d * 0.05; }
    else if (d < 0.35) { color = mix(vec3(0.1,0.0,0.0), vec3(0.5,0.05,0.0), (d-0.15)/0.2); a = 0.03 + (d-0.15)*0.15; }
    else if (d < 0.55) { color = mix(vec3(0.5,0.05,0.0), vec3(1.0,0.3,0.0), (d-0.35)/0.2); a = 0.06 + (d-0.35)*0.35; }
    else if (d < 0.75) { color = mix(vec3(1.0,0.3,0.0), vec3(1.0,0.8,0.1), (d-0.55)/0.2); a = 0.13 + (d-0.55)*0.6; }
    else { color = mix(vec3(1.0,0.8,0.1), vec3(1.0,1.0,1.0), (d-0.75)/0.25); a = 0.25 + (d-0.75)*2.5; }
    return vec4(color, a);
}

vec4 tfCool(float d) {
    vec3 color; float a;
    if (d < 0.2) { color = vec3(0.0, 0.0, 0.05); a = d * 0.04; }
    else if (d < 0.4) { color = mix(vec3(0.0,0.0,0.2), vec3(0.0,0.3,0.6), (d-0.2)/0.2); a = 0.03 + (d-0.2)*0.2; }
    else if (d < 0.6) { color = mix(vec3(0.0,0.3,0.6), vec3(0.2,0.7,0.9), (d-0.4)/0.2); a = 0.07 + (d-0.4)*0.4; }
    else if (d < 0.8) { color = mix(vec3(0.2,0.7,0.9), vec3(0.6,0.9,1.0), (d-0.6)/0.2); a = 0.15 + (d-0.6)*0.7; }
    else { color = mix(vec3(0.6,0.9,1.0), vec3(1.0,1.0,1.0), (d-0.8)/0.2); a = 0.3 + (d-0.8)*2.0; }
    return vec4(color, a);
}

vec4 tfFire(float d) {
    vec3 color; float a;
    if (d < 0.2) { color = vec3(0.05,0.0,0.0); a = d * 0.06; }
    else if (d < 0.4) { color = mix(vec3(0.3,0.0,0.0), vec3(0.8,0.1,0.0), (d-0.2)/0.2); a = 0.04 + (d-0.2)*0.25; }
    else if (d < 0.6) { color = mix(vec3(0.8,0.1,0.0), vec3(1.0,0.5,0.0), (d-0.4)/0.2); a = 0.09 + (d-0.4)*0.5; }
    else if (d < 0.8) { color = mix(vec3(1.0,0.5,0.0), vec3(1.0,0.9,0.2), (d-0.6)/0.2); a = 0.19 + (d-0.6)*0.8; }
    else { color = mix(vec3(1.0,0.9,0.2), vec3(1.0,1.0,1.0), (d-0.8)/0.2); a = 0.35 + (d-0.8)*1.8; }
    return vec4(color, a);
}

vec4 tfRainbow(float d) {
    vec3 color; float a;
    if (d < 0.17) { color = vec3(0.0,0.0,0.3); a = d*0.05; }
    else if (d < 0.33) { color = mix(vec3(0.0,0.0,0.8), vec3(0.0,0.8,0.8), (d-0.17)/0.16); a = 0.04+(d-0.17)*0.2; }
    else if (d < 0.50) { color = mix(vec3(0.0,0.8,0.8), vec3(0.0,0.8,0.0), (d-0.33)/0.17); a = 0.07+(d-0.33)*0.3; }
    else if (d < 0.67) { color = mix(vec3(0.0,0.8,0.0), vec3(0.8,0.8,0.0), (d-0.50)/0.17); a = 0.12+(d-0.50)*0.5; }
    else if (d < 0.83) { color = mix(vec3(0.8,0.8,0.0), vec3(1.0,0.4,0.0), (d-0.67)/0.16); a = 0.20+(d-0.67)*0.7; }
    else { color = mix(vec3(1.0,0.4,0.0), vec3(0.8,0.0,0.2), (d-0.83)/0.17); a = 0.32+(d-0.83)*2.0; }
    return vec4(color, a);
}

// yt-style log-space layered transfer function
vec4 tfYT(float density) {
    // Recover original density: uint8 [0,1] → raw [8,14]
    float raw = density * 6.0 + 8.0;
    // Log10 mapping (yt set_log=true)
    float logD = log(raw) / log(10.0);
    // Normalize to [0,1] based on log10(8)=0.903, log10(14)=1.146
    float t = clamp((logD - 0.903) / 0.243, 0.0, 1.0);
    
    // 5-layer transfer function with hot colormap (yt add_layers 5)
    vec3 color; float a;
    if (t < 0.20) { color = vec3(0.0, 0.0, 0.02); a = 0.05; }
    else if (t < 0.40) { color = mix(vec3(0.1,0.0,0.0), vec3(0.5,0.05,0.0), (t-0.20)/0.20); a = 0.15; }
    else if (t < 0.60) { color = mix(vec3(0.5,0.05,0.0), vec3(1.0,0.3,0.0), (t-0.40)/0.20); a = 0.35; }
    else if (t < 0.80) { color = mix(vec3(1.0,0.3,0.0), vec3(1.0,0.8,0.1), (t-0.60)/0.20); a = 0.60; }
    else { color = mix(vec3(1.0,0.8,0.1), vec3(1.0,1.0,1.0), (t-0.80)/0.20); a = 0.85; }
    
    // Gaussian layer profile (yt-style layer blending)
    float layerW = 0.20;
    float layerCenter = floor(t / layerW) * layerW + layerW * 0.5;
    float sigma = layerW * 0.40;
    float gauss = exp(-pow((t - layerCenter) / sigma, 2.0));
    a *= gauss;
    
    return vec4(color * uBrightness, a * uOpacityScale);
}

float logNormalize(float d) {
    if (uLogScale != 1) return d;
    float rmin = max(uRawMin, 0.001);
    float raw = d * (uRawMax - rmin) + rmin;
    float logD = log(raw) / log(10.0);
    float logMin = log(rmin) / log(10.0);
    float logMax = log(uRawMax) / log(10.0);
    return clamp((logD - logMin) / (logMax - logMin), 0.0, 1.0);
}

vec4 transferFunction(float density) {
    float d = density;
    vec4 result;
    if (uTFMode == 1) result = tfCool(d);
    else if (uTFMode == 2) result = tfFire(d);
    else if (uTFMode == 3) result = tfRainbow(d);
    else if (uTFMode == 4) return tfYT(density);
    else result = tfHot(d);
    result.a *= uOpacityScale;
    result.a = clamp(result.a, 0.0, 1.0);
    return vec4(result.rgb * uBrightness, result.a);
}

void main() {
    vec3 rayDir = normalize(vWorldPos - uCameraPos);
    vec2 tNearFar = rayBoxIntersection(uCameraPos, rayDir, uBoxMin, uBoxMax);
    float tNear = tNearFar.x;
    float tFar = tNearFar.y;
    if (tNear > tFar || tFar < 0.0) discard;
    tNear = max(tNear, 0.0);
    float distance = tFar - tNear;
    if (distance <= 0.0) discard;
    vec3 startPos = uCameraPos + rayDir * tNear;
    float stepSize = uStepSize;
    int maxSteps = int(distance / stepSize) + 1;
    if (maxSteps > 256) maxSteps = 256;
    float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    vec3 currentPos = startPos + rayDir * stepSize * jitter;
    vec3 accumColor = vec3(0.0);
    float accumAlpha = 0.0;
    for (int i = 0; i < 256; i++) {
        if (i >= maxSteps || accumAlpha > 0.97) break;
        vec3 texCoord = (currentPos - uBoxMin) / (uBoxMax - uBoxMin);
        texCoord = clamp(texCoord, 0.0, 1.0);
        float rawDensity = texture(uVolume, texCoord).r;
        if (rawDensity < uThresholdMin || rawDensity > uThresholdMax) {
            currentPos += rayDir * stepSize;
            continue;
        }
        vec4 sampleColor = transferFunction(logNormalize(rawDensity));
        accumColor += sampleColor.rgb * sampleColor.a * (1.0 - accumAlpha);
        accumAlpha += sampleColor.a * (1.0 - accumAlpha);
        currentPos += rayDir * stepSize;
    }
    if (accumAlpha < 0.01) discard;
    gl_FragColor = vec4(accumColor, accumAlpha);
}
`;

function clearContainer(id) { const el = document.getElementById(id); if (el) el.innerHTML = ''; }

function setLoadingProgress(done, total) {
    const bar = document.getElementById('loading-bar');
    const detail = document.getElementById('loading-detail');
    if (bar) bar.style.width = (done / total * 100).toFixed(0) + '%';
    if (detail) detail.innerText = `${done} / ${total}`;
}

function hideLoadingScreen() {
    const screen = document.getElementById('loading-screen');
    if (screen) {
        screen.classList.add('hidden');
        setTimeout(() => screen.style.display = 'none', 600);
    }
}

// ===== Preload all data =====
async function preloadAllData() {
    const [sRes, hRes] = await Promise.all([
        fetch('/static/data/stats.json'),
        fetch('/static/data/histograms.json')
    ]);
    statsData = await sRes.json();
    histogramsData = await hRes.json();

    const BATCH_SIZE = 10;
    for (let batch = 0; batch < 100; batch += BATCH_SIZE) {
        const promises = [];
        for (let ts = batch; ts < Math.min(batch + BATCH_SIZE, 100); ts++) {
            promises.push(
                fetch(`/static/data/volumes/vol_${String(ts).padStart(4, '0')}.bin`)
                    .then(r => r.arrayBuffer())
                    .then(buf => {
                        volumeData[ts] = new Uint8Array(buf);
                        volumeLoaded++;
                        if (volumeLoaded % 5 === 0) setLoadingProgress(volumeLoaded, 100);
                    })
                    .catch(e => console.warn(`Failed to load volume ${ts}:`, e))
            );
        }
        await Promise.all(promises);
    }
    setLoadingProgress(100, 100);
}

// ===== Main Update =====
function onTimestepDrag() {
    const val = parseInt(document.getElementById('ctrl-timestep').value);
    currentTs = val;
    document.getElementById('ctrl-ts-label').innerText = val;
    document.getElementById('vol-badge').innerText = 't=' + val;

    updateProjectionsInteractive(val);
    updateStatPlotsInstant(val);
    updateVolume3DInstant(val);
    updateFilterInfo();
}

function onProjectionTypeChange() {
    currentProjType = document.getElementById('ctrl-proj').value;
    updateProjectionsInteractive(currentTs);
}

// ===== Interactive Projections: Canvas + Tooltip =====
async function updateProjectionsInteractive(ts) {
    const type = currentProjType;

    if (abortCtrl) abortCtrl.abort();
    abortCtrl = new AbortController();
    const signal = abortCtrl.signal;

    try {
        const [rx, ry, rz] = await Promise.all([
            getProjection(ts, 0, type, signal),
            getProjection(ts, 1, type, signal),
            getProjection(ts, 2, type, signal)
        ]);
        if (signal.aborted) return;

        drawHeatmapInteractive('vol-proj-x', rx.projection, rx.min, rx.max, `XY 投影 (t=${ts})`);
        drawHeatmapInteractive('vol-proj-y', ry.projection, ry.min, ry.max, `YZ 投影 (t=${ts})`);
        drawHeatmapInteractive('vol-proj-z', rz.projection, rz.min, rz.max, `XZ 投影 (t=${ts})`);
    } catch (e) {
        if (e.name !== 'AbortError') console.error('Projection load failed:', e);
    }
}

async function getProjection(ts, axis, type, signal) {
    const key = `${ts}_${axis}_${type}`;
    if (projCache.has(key)) return projCache.get(key);
    const resp = await fetch(`/api/timestep/${ts}/projection?axis=${axis}&type=${type}`, { signal });
    const data = await resp.json();
    projCache.set(key, data);
    return data;
}

function drawHeatmapInteractive(containerId, matrix, minVal, maxVal, title) {
    const container = document.getElementById(containerId);
    const canvas = container.querySelector('canvas');
    const tooltip = container.querySelector('.tooltip');
    if (!canvas || !tooltip) return;

    const h = matrix.length;
    const w = matrix[0].length;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(w, h);

    const range = (maxVal - minVal) || 1;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const t = Math.max(0, Math.min(1, (matrix[y][x] - minVal) / range));
            const [r, g, b] = hotColor(t);
            const idx = (y * w + x) * 4;
            imgData.data[idx] = r;
            imgData.data[idx + 1] = g;
            imgData.data[idx + 2] = b;
            imgData.data[idx + 3] = 255;
        }
    }
    ctx.putImageData(imgData, 0, 0);

    // Store matrix data on canvas for tooltip
    canvas._matrix = matrix;
    canvas._title = title;

    // Remove old listeners by replacing canvas with fresh element (but redraw!)
    const newCanvas = document.createElement('canvas');
    newCanvas.width = w;
    newCanvas.height = h;
    newCanvas.style.cssText = canvas.style.cssText;
    const newCtx = newCanvas.getContext('2d');
    newCtx.putImageData(imgData, 0, 0);
    newCanvas._matrix = matrix;
    newCanvas._title = title;
    container.replaceChild(newCanvas, canvas);

    newCanvas.addEventListener('mousemove', (e) => {
        const rect = newCanvas.getBoundingClientRect();
        const scaleX = newCanvas.width / rect.width;
        const scaleY = newCanvas.height / rect.height;
        const x = Math.floor((e.clientX - rect.left) * scaleX);
        const y = Math.floor((e.clientY - rect.top) * scaleY);
        const mat = newCanvas._matrix;
        const ttl = newCanvas._title;

        if (mat && y >= 0 && y < mat.length && x >= 0 && x < mat[0].length) {
            const val = mat[y][x];
            tooltip.innerHTML = `<b>${ttl}</b><br>x:${x} y:${y}<br>密度: ${val.toFixed(3)}`;
            tooltip.style.left = Math.min(e.clientX - rect.left + 12, rect.width - 120) + 'px';
            tooltip.style.top = Math.min(e.clientY - rect.top + 12, rect.height - 50) + 'px';
            tooltip.classList.add('visible');
        }
    });

    newCanvas.addEventListener('mouseleave', () => {
        tooltip.classList.remove('visible');
    });
}

// ===== Stat Plots: INSTANT Plotly from memory =====
function updateStatPlotsInstant(ts) {
    const h = histogramsData[ts];
    if (!h) return;
    renderStatSummary(h);
    // Throttle: cancel pending RAF and schedule new one
    if (statRafId) cancelAnimationFrame(statRafId);
    statRafId = requestAnimationFrame(() => {
        statRafId = null;
        renderStatHistograms(h);
        renderStatCDF(h);
    });
}

function renderStatSummary(data) {
    const s = data.stats;
    const el = document.getElementById('stat-summary');
    if (!el) return;
    const items = [
        { label: '均值', val: s.mean.toFixed(3) },
        { label: '标准差', val: s.std.toFixed(3) },
        { label: '最小值', val: s.min.toFixed(2) },
        { label: '最大值', val: s.max.toFixed(2) },
        { label: '中位数', val: s.median.toFixed(3) },
        { label: 'P1', val: s.p1.toFixed(2) },
        { label: 'P95', val: s.p95.toFixed(2) },
        { label: 'P99', val: s.p99.toFixed(2) },
    ];
    el.innerHTML = items.map(it => `<div class="s-item">${it.label}<b>${it.val}</b></div>`).join('');
}

// Pre-computed static layouts (shared across all timesteps)
const HIST_LAYOUT = {
    paper_bgcolor: PLOT_PAPER, plot_bgcolor: PLOT_BG, font: { color: PLOT_MUTED, size: 10 },
    title: { text: '密度分布直方图（框选可筛选）', font: { color: PLOT_TEXT, size: 12 } },
    xaxis: { title: { text: '密度值 ρ', font: { size: 10 } }, gridcolor: PLOT_GRID, color: PLOT_MUTED, tickfont: { size: 9 } },
    yaxis: { title: { text: '频数', font: { size: 10 } }, gridcolor: PLOT_GRID, color: PLOT_MUTED, type: 'linear', tickfont: { size: 9 } },
    margin: { t: 28, b: 36, l: 44, r: 10 },
    hovermode: 'closest',
    dragmode: 'select',
    selectdirection: 'h'
};
const LOGHIST_LAYOUT = {
    paper_bgcolor: PLOT_PAPER, plot_bgcolor: PLOT_BG, font: { color: PLOT_MUTED, size: 10 },
    title: { text: '密度对数分布 log₁₀(ρ)（框选可筛选）', font: { color: PLOT_TEXT, size: 12 } },
    xaxis: { title: { text: 'log₁₀(ρ)', font: { size: 10 } }, gridcolor: PLOT_GRID, color: PLOT_MUTED, tickfont: { size: 9 } },
    yaxis: { title: { text: '频数', font: { size: 10 } }, gridcolor: PLOT_GRID, color: PLOT_MUTED, tickfont: { size: 9 } },
    margin: { t: 28, b: 36, l: 44, r: 10 },
    hovermode: 'closest',
    dragmode: 'select',
    selectdirection: 'h'
};
const CDF_LAYOUT_BASE = {
    paper_bgcolor: PLOT_PAPER, plot_bgcolor: PLOT_BG, font: { color: PLOT_MUTED, size: 10 },
    title: { text: '密度累积分布函数 (CDF)', font: { color: PLOT_TEXT, size: 12 } },
    xaxis: { title: { text: '密度值', font: { size: 10 } }, gridcolor: PLOT_GRID, color: PLOT_MUTED, tickfont: { size: 9 } },
    yaxis: { title: { text: '累积概率', font: { size: 10 } }, gridcolor: PLOT_GRID, color: PLOT_MUTED, range: [0, 1], tickfont: { size: 9 } },
    margin: { t: 28, b: 36, l: 44, r: 10 },
    hovermode: 'closest',
    dragmode: 'select',
    selectdirection: 'h'
};

function renderStatHistograms(data) {
    const edges = data.edges;
    const centers = edges.slice(0, -1).map((e, i) => (e + edges[i+1]) / 2);
    const logEdges = data.log_edges;
    const logCenters = logEdges.slice(0, -1).map((e, i) => (e + logEdges[i+1]) / 2);

    // 构建筛选区间 shapes
    let filterShapes = [];
    let logFilterShapes = [];
    if (densityFilterRaw.min !== null || densityFilterRaw.max !== null) {
        const s = data.stats;
        const fmin = densityFilterRaw.min !== null ? densityFilterRaw.min : s.min;
        const fmax = densityFilterRaw.max !== null ? densityFilterRaw.max : s.max;
        filterShapes = [{
            type: 'rect',
            x0: fmin, x1: fmax,
            y0: 0, y1: 1,
            yref: 'paper',
            fillcolor: 'rgba(79, 172, 254, 0.18)',
            line: { color: '#4facfe', width: 1.5, dash: 'dot' },
            layer: 'below'
        }];
        logFilterShapes = [{
            type: 'rect',
            x0: Math.log10(fmin), x1: Math.log10(fmax),
            y0: 0, y1: 1,
            yref: 'paper',
            fillcolor: 'rgba(79, 172, 254, 0.18)',
            line: { color: '#4facfe', width: 1.5, dash: 'dot' },
            layer: 'below'
        }];
    }

    const trace1 = {
        x: centers, y: data.hist, type: 'bar',
        marker: { color: 'rgba(79, 172, 254, 0.7)', line: { color: 'rgba(79, 172, 254, 1)', width: 0.5 } },
        hovertemplate: '密度: %{x:.3f}<br>频数: %{y}<extra></extra>'
    };
    const trace2 = {
        x: logCenters, y: data.log_hist, type: 'bar',
        marker: { color: 'rgba(0, 242, 254, 0.6)' },
        hovertemplate: 'log₁₀(ρ): %{x:.4f}<br>频数: %{y}<extra></extra>'
    };

    const histLayout = Object.assign({}, HIST_LAYOUT, { shapes: filterShapes });
    const logHistLayout = Object.assign({}, LOGHIST_LAYOUT, { shapes: logFilterShapes });

    if (!statsInitialized) {
        Plotly.newPlot('stat-hist-plot', [trace1], histLayout, { responsive: true, displayModeBar: false });
        Plotly.newPlot('stat-loghist-plot', [trace2], logHistLayout, { responsive: true, displayModeBar: false });
        enableHistSelection();
    } else {
        Plotly.react('stat-hist-plot', [trace1], histLayout, { responsive: true, displayModeBar: false });
        Plotly.react('stat-loghist-plot', [trace2], logHistLayout, { responsive: true, displayModeBar: false });
    }
}

function renderStatCDF(data) {
    const edges = data.edges;
    const centers = edges.slice(0, -1).map((e, i) => (e + edges[i+1]) / 2);
    let cum = 0;
    const cdf = data.hist.map(v => { cum += v; return cum; });
    const total = cum;
    const cdfNorm = cdf.map(v => v / total);

    const trace = {
        x: centers, y: cdfNorm, type: 'scatter', mode: 'lines',
        line: { color: '#a18cd1', width: 2 },
        fill: 'tozeroy', fillcolor: 'rgba(161, 140, 209, 0.12)',
        hovertemplate: '密度: %{x:.3f}<br>累积概率: %{y:.4f}<extra></extra>'
    };

    function findCdfY(pct) {
        const target = pct / 100.0;
        for (let i = 0; i < cdfNorm.length; i++) {
            if (cdfNorm[i] >= target) return cdfNorm[i];
        }
        return target;
    }

    const s = data.stats;
    const yP1 = findCdfY(1), yP5 = findCdfY(5), yP95 = findCdfY(95), yP99 = findCdfY(99);

    const layout = Object.assign({}, CDF_LAYOUT_BASE, {
        shapes: [
            { type: 'line', x0: s.p1, x1: s.p1, y0: 0, y1: yP1, line: { color: '#51cf66', width: 1, dash: 'dot' } },
            { type: 'line', x0: s.p5, x1: s.p5, y0: 0, y1: yP5, line: { color: '#51cf66', width: 1, dash: 'dot' } },
            { type: 'line', x0: s.p95, x1: s.p95, y0: 0, y1: yP95, line: { color: '#ff6b6b', width: 1, dash: 'dot' } },
            { type: 'line', x0: s.p99, x1: s.p99, y0: 0, y1: yP99, line: { color: '#ff6b6b', width: 1, dash: 'dot' } }
        ],
        annotations: [
            { x: s.p1, y: yP1 + 0.03, text: 'P1', showarrow: false, font: { color: '#51cf66', size: 9 } },
            { x: s.p5, y: yP5 + 0.03, text: 'P5', showarrow: false, font: { color: '#51cf66', size: 9 } },
            { x: s.p95, y: yP95 - 0.05, text: 'P95', showarrow: false, font: { color: '#ff6b6b', size: 9 } },
            { x: s.p99, y: yP99 - 0.05, text: 'P99', showarrow: false, font: { color: '#ff6b6b', size: 9 } }
        ]
    });

    if (!statsInitialized) {
        Plotly.newPlot('stat-cdf-plot', [trace], layout, { responsive: true, displayModeBar: false });
        statsInitialized = true;
    } else {
        Plotly.react('stat-cdf-plot', [trace], layout, { responsive: true, displayModeBar: false });
    }
}

// ===== Volume 3D Renderer with high-performance GPU =====
function initVolume3D() {
    if (vol3DInitialized) return;
    const container = document.getElementById('vol-three-container');
    if (!container) return;

    const testCanvas = document.createElement('canvas');
    const gl2 = testCanvas.getContext('webgl2', { powerPreference: 'high-performance' });
    const hasRM = gl2 && typeof THREE.DataTexture3D !== 'undefined';

    volScene = new THREE.Scene();
    volScene.background = new THREE.Color(0x050610);

    volCamera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 100);
    volCamera.position.set(2.8, 2.2, 2.8);

    volRenderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: 'high-performance'
    });
    volRenderer.setSize(container.clientWidth, container.clientHeight);
    volRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(volRenderer.domElement);

    volControls = new THREE.OrbitControls(volCamera, volRenderer.domElement);
    volControls.enableDamping = true;
    volControls.dampingFactor = 0.05;
    volControls.autoRotate = true;
    volControls.autoRotateSpeed = 0.3;
    volControls.target.set(0, 0, 0);

    const boxGeo = new THREE.BoxGeometry(2, 2, 2);
    const boxEdges = new THREE.EdgesGeometry(boxGeo);
    volScene.add(new THREE.LineSegments(boxEdges, new THREE.LineBasicMaterial({ color: 0x3a3a55 })));

    if (hasRM) {
        volRayMaterial = new THREE.ShaderMaterial({
            vertexShader: RM_VERT, fragmentShader: RM_FRAG,
            uniforms: {
                uCameraPos: { value: new THREE.Vector3() },
                uBoxMin: { value: new THREE.Vector3(-1, -1, -1) },
                uBoxMax: { value: new THREE.Vector3(1, 1, 1) },
                uVolume: { value: null }, uStepSize: { value: 2.0 / 64.0 },
                uOpacityScale: { value: 1.2 }, uThresholdMin: { value: 0.0 },
                uThresholdMax: { value: 1.0 }, uBrightness: { value: 1.3 },
                uTFMode: { value: 0 },
                uRawMin: { value: 8.0 }, uRawMax: { value: 14.0 },
                uLogScale: { value: 0 }
            },
            transparent: true, depthWrite: true, side: THREE.DoubleSide
        });
        volMesh = new THREE.Mesh(boxGeo, volRayMaterial);
        volScene.add(volMesh);
    } else {
        document.getElementById('vol-three-overlay').innerText = '浏览器不支持 WebGL 2.0 3D 纹理';
    }

    function animateVol() {
        requestAnimationFrame(animateVol);
        if (volControls) volControls.update();
        if (volRayMaterial) volRayMaterial.uniforms.uCameraPos.value.copy(volCamera.position);
        if (volRenderer && volScene && volCamera) volRenderer.render(volScene, volCamera);
    }
    animateVol();

    window.addEventListener('resize', () => {
        if (!volCamera || !volRenderer) return;
        const c = document.getElementById('vol-three-container');
        if (!c) return;
        volCamera.aspect = c.clientWidth / c.clientHeight;
        volCamera.updateProjectionMatrix();
        volRenderer.setSize(c.clientWidth, c.clientHeight);
    });

    vol3DInitialized = true;
}

function updateVolume3DInstant(ts) {
    if (!volRayMaterial) return;
    const bytes = volumeData[ts];
    if (!bytes) {
        document.getElementById('vol-three-overlay').innerText = '体数据未加载';
        return;
    }

    const [w, h, d] = volMeta.shape;
    if (volTexture) volTexture.dispose();

    volTexture = new THREE.DataTexture3D(bytes, w, h, d);
    volTexture.format = THREE.RedFormat;
    volTexture.type = THREE.UnsignedByteType;
    volTexture.minFilter = THREE.LinearFilter;
    volTexture.magFilter = THREE.LinearFilter;
    volTexture.wrapS = THREE.ClampToEdgeWrapping;
    volTexture.wrapT = THREE.ClampToEdgeWrapping;
    volTexture.wrapR = THREE.ClampToEdgeWrapping;
    volTexture.needsUpdate = true;

    volRayMaterial.uniforms.uVolume.value = volTexture;
    volRayMaterial.uniforms.uStepSize.value = 2.0 / w;
    const rmin = volMeta.raw_min > 0.001 ? volMeta.raw_min : 8.0;
    const rmax = volMeta.raw_max > rmin ? volMeta.raw_max : 14.0;
    volRayMaterial.uniforms.uRawMin.value = rmin;
    volRayMaterial.uniforms.uRawMax.value = rmax;
    document.getElementById('vol-three-overlay').innerText = '';

    updateVolume3DUniforms();
}

function updateVolume3DUniforms() {
    if (!volRayMaterial) return;
    const opacity = parseFloat(document.getElementById('ctrl-opacity').value);
    const brightness = parseFloat(document.getElementById('ctrl-brightness').value);
    const tfMode = parseInt(document.getElementById('ctrl-tf').value);
    const quality = parseInt(document.getElementById('ctrl-quality').value);
    const logScaleEl = document.getElementById('ctrl-log');
    const logScale = logScaleEl && logScaleEl.checked ? 1 : 0;

    document.getElementById('ctrl-opacity-val').innerText = opacity.toFixed(1);
    document.getElementById('ctrl-brightness-val').innerText = brightness.toFixed(1);

    const stepSizes = [4.0/64.0, 2.0/64.0, 1.0/64.0];
    volRayMaterial.uniforms.uStepSize.value = stepSizes[quality] || (2.0/64.0);
    volRayMaterial.uniforms.uOpacityScale.value = opacity;
    volRayMaterial.uniforms.uBrightness.value = brightness;
    volRayMaterial.uniforms.uTFMode.value = tfMode;
    volRayMaterial.uniforms.uLogScale.value = logScale;

    applyDensityFilter();
    drawTFCurve();
}

// ===== Density Filter (相空间联动筛选) =====
function applyDensityFilter() {
    if (!volRayMaterial) return;
    const s = statsData[currentTs];
    if (!s) return;
    let tMin = 0.0, tMax = 1.0;
    if (densityFilterRaw.min !== null) {
        tMin = (densityFilterRaw.min - s.min) / (s.max - s.min);
    }
    if (densityFilterRaw.max !== null) {
        tMax = (densityFilterRaw.max - s.min) / (s.max - s.min);
    }
    volRayMaterial.uniforms.uThresholdMin.value = Math.max(0, Math.min(1, tMin));
    volRayMaterial.uniforms.uThresholdMax.value = Math.max(0, Math.min(1, tMax));
}

function computePercentileFromHist(hist, edges, p) {
    const total = hist.reduce((a, b) => a + b, 0);
    const target = total * p / 100;
    let cum = 0;
    for (let i = 0; i < hist.length; i++) {
        cum += hist[i];
        if (cum >= target) {
            const prevCum = cum - hist[i];
            const frac = hist[i] > 0 ? (target - prevCum) / hist[i] : 0;
            return edges[i] + frac * (edges[i+1] - edges[i]);
        }
    }
    return edges[edges.length - 1];
}

function setFilterPreset(preset) {
    const s = statsData[currentTs];
    const h = histogramsData[currentTs];
    if (!s || !h) return;
    switch(preset) {
        case 'all':
            densityFilterRaw = { min: null, max: null };
            break;
        case 'top1':
            densityFilterRaw = { min: s.p99, max: s.max };
            break;
        case 'top5':
            densityFilterRaw = { min: s.p95, max: s.max };
            break;
        case 'top10': {
            const p90 = computePercentileFromHist(h.hist, h.edges, 90);
            densityFilterRaw = { min: p90, max: s.max };
            break;
        }
    }
    applyDensityFilter();
    updateFilterInfo();
    updateCustomFilterInputs();
    const hd = histogramsData[currentTs];
    if (hd) {
        renderStatHistograms(hd);
        renderStatCDF(hd);
    }
}

function applyCustomFilter() {
    const minInput = document.getElementById('filter-min');
    const maxInput = document.getElementById('filter-max');
    const rawMin = parseFloat(minInput.value);
    const rawMax = parseFloat(maxInput.value);
    const s = statsData[currentTs];
    if (!s) return;
    densityFilterRaw = {
        min: (!isNaN(rawMin) && rawMin > 0) ? rawMin : null,
        max: (!isNaN(rawMax) && rawMax > 0) ? rawMax : null
    };
    applyDensityFilter();
    updateFilterInfo();
    const hd = histogramsData[currentTs];
    if (hd) {
        renderStatHistograms(hd);
        renderStatCDF(hd);
    }
}

function updateCustomFilterInputs() {
    const minInput = document.getElementById('filter-min');
    const maxInput = document.getElementById('filter-max');
    if (!minInput || !maxInput) return;
    if (densityFilterRaw.min !== null) minInput.value = densityFilterRaw.min.toFixed(2);
    else minInput.value = '';
    if (densityFilterRaw.max !== null) maxInput.value = densityFilterRaw.max.toFixed(2);
    else maxInput.value = '';
}

function applyBrushFilter(rawMin, rawMax) {
    densityFilterRaw = { min: rawMin, max: rawMax };
    applyDensityFilter();
    updateFilterInfo();
    updateCustomFilterInputs();
    const hd = histogramsData[currentTs];
    if (hd) {
        renderStatHistograms(hd);
        renderStatCDF(hd);
    }
}

let histSelectEnabled = false;
function enableHistSelection() {
    if (histSelectEnabled) return;
    histSelectEnabled = true;
    const histEl = document.getElementById('stat-hist-plot');
    const logEl = document.getElementById('stat-loghist-plot');
    if (histEl) {
        histEl.on('plotly_selected', (eventData) => {
            if (!eventData || !eventData.range) return;
            const xRange = eventData.range.x;
            if (!xRange || xRange.length < 2) return;
            applyBrushFilter(xRange[0], xRange[1]);
        });
    }
    if (logEl) {
        logEl.on('plotly_selected', (eventData) => {
            if (!eventData || !eventData.range) return;
            const xRange = eventData.range.x;
            if (!xRange || xRange.length < 2) return;
            // log-scale selection: convert back from log10
            const rawMin = Math.pow(10, xRange[0]);
            const rawMax = Math.pow(10, xRange[1]);
            applyBrushFilter(rawMin, rawMax);
        });
    }
}

function updateFilterInfo() {
    const s = statsData[currentTs];
    const h = histogramsData[currentTs];
    const elStatus = document.getElementById('filter-status');
    const elRawRange = document.getElementById('filter-raw-range');
    const elVoxelRatio = document.getElementById('filter-voxel-ratio');
    if (!s || !h) return;
    if (densityFilterRaw.min === null && densityFilterRaw.max === null) {
        if (elStatus) elStatus.style.display = 'none';
        return;
    }
    const minVal = densityFilterRaw.min !== null ? densityFilterRaw.min : s.min;
    const maxVal = densityFilterRaw.max !== null ? densityFilterRaw.max : s.max;
    if (elRawRange) elRawRange.innerText = `[${minVal.toFixed(2)}, ${maxVal.toFixed(2)}]`;
    let selectedCount = 0;
    const edges = h.edges;
    for (let i = 0; i < h.hist.length; i++) {
        const binMin = edges[i];
        const binMax = edges[i+1];
        if (binMax >= minVal && binMin <= maxVal) {
            const overlapMin = Math.max(binMin, minVal);
            const overlapMax = Math.min(binMax, maxVal);
            const overlapRatio = Math.max(0, overlapMax - overlapMin) / (binMax - binMin);
            selectedCount += h.hist[i] * overlapRatio;
        }
    }
    const total = h.hist.reduce((a, b) => a + b, 0);
    const ratio = (selectedCount / total * 100).toFixed(2);
    if (elVoxelRatio) elVoxelRatio.innerText = ratio + '%';
    if (elStatus) elStatus.style.display = 'block';
}

function drawTFCurve() {
    const canvas = document.getElementById('vol-tf-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    ctx.fillRect(0, 0, w, h);

    const tfMode = parseInt(document.getElementById('ctrl-tf').value);
    const opacity = parseFloat(document.getElementById('ctrl-opacity').value);
    const brightness = parseFloat(document.getElementById('ctrl-brightness').value);

    for (let x = 0; x < w; x++) {
        const d = x / w;
        let r, g, b;
        if (tfMode == 1) {
            if (d < 0.2) { r=0; g=0; b=0.05; } else if (d < 0.4) { r=0; g=0.3; b=0.6; } else if (d < 0.6) { r=0.2; g=0.7; b=0.9; } else if (d < 0.8) { r=0.6; g=0.9; b=1.0; } else { r=1.0; g=1.0; b=1.0; }
        } else if (tfMode == 2) {
            if (d < 0.2) { r=0.3; g=0; b=0; } else if (d < 0.4) { r=0.8; g=0.1; b=0; } else if (d < 0.6) { r=1.0; g=0.5; b=0; } else if (d < 0.8) { r=1.0; g=0.9; b=0.2; } else { r=1.0; g=1.0; b=1.0; }
        } else if (tfMode == 3) {
            if (d < 0.17) { r=0; g=0; b=0.8; } else if (d < 0.33) { r=0; g=0.8; b=0.8; } else if (d < 0.50) { r=0; g=0.8; b=0; } else if (d < 0.67) { r=0.8; g=0.8; b=0; } else if (d < 0.83) { r=1.0; g=0.4; b=0; } else { r=0.8; g=0; b=0.2; }
        } else if (tfMode == 4) {
            // yt Log-Layers: log-space hot colormap (5 layers)
            let ct = d;
            if (ct < 0.20) { r=0.0; g=0.0; b=0.02; }
            else if (ct < 0.40) { r=0.1+(0.5-0.1)*((ct-0.2)/0.2); g=0.0+(0.05-0.0)*((ct-0.2)/0.2); b=0.0; }
            else if (ct < 0.60) { r=0.5+(1.0-0.5)*((ct-0.4)/0.2); g=0.05+(0.3-0.05)*((ct-0.4)/0.2); b=0.0; }
            else if (ct < 0.80) { r=1.0+(1.0-1.0)*((ct-0.6)/0.2); g=0.3+(0.8-0.3)*((ct-0.6)/0.2); b=0.0+(0.1-0.0)*((ct-0.6)/0.2); }
            else { r=1.0+(1.0-1.0)*((ct-0.8)/0.2); g=0.8+(1.0-0.8)*((ct-0.8)/0.2); b=0.1+(1.0-0.1)*((ct-0.8)/0.2); }
        } else {
            if (d < 0.15) { r=0; g=0; b=0.02; } else if (d < 0.35) { r=0.5; g=0.05; b=0; } else if (d < 0.55) { r=1.0; g=0.3; b=0; } else if (d < 0.75) { r=1.0; g=0.8; b=0.1; } else { r=1.0; g=1.0; b=1.0; }
        }
        ctx.fillStyle = `rgb(${Math.min(255, r*255*brightness)}, ${Math.min(255, g*255*brightness)}, ${Math.min(255, b*255*brightness)})`;
        ctx.fillRect(x, h/2, 1, h/2);
    }

    ctx.strokeStyle = '#4facfe';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let x = 0; x < w; x++) {
        const d = x / w;
        let a;
        if (tfMode == 1) {
            if (d < 0.2) a = d * 0.04; else if (d < 0.4) a = 0.03 + (d-0.2)*0.2; else if (d < 0.6) a = 0.07 + (d-0.4)*0.4; else if (d < 0.8) a = 0.15 + (d-0.6)*0.7; else a = 0.3 + (d-0.8)*2.0;
        } else if (tfMode == 2) {
            if (d < 0.2) a = d * 0.06; else if (d < 0.4) a = 0.04 + (d-0.2)*0.25; else if (d < 0.6) a = 0.09 + (d-0.4)*0.5; else if (d < 0.8) a = 0.19 + (d-0.6)*0.8; else a = 0.35 + (d-0.8)*1.8;
        } else if (tfMode == 3) {
            if (d < 0.17) a = d*0.05; else if (d < 0.33) a = 0.04+(d-0.17)*0.2; else if (d < 0.50) a = 0.07+(d-0.33)*0.3; else if (d < 0.67) a = 0.12+(d-0.50)*0.5; else if (d < 0.83) a = 0.20+(d-0.67)*0.7; else a = 0.32+(d-0.83)*2.0;
        } else if (tfMode == 4) {
            // yt: 5 Gaussian layers
            a = 0.0;
            for (let li = 0; li < 5; li++) {
                let lc = (li + 0.5) * 0.20;
                let sigma = 0.20 * 0.40;
                let la = [0.05, 0.15, 0.35, 0.60, 0.85][li];
                a += la * Math.exp(-Math.pow((d - lc) / sigma, 2));
            }
        } else {
            if (d < 0.15) a = d * 0.05; else if (d < 0.35) a = 0.03 + (d-0.15)*0.15; else if (d < 0.55) a = 0.06 + (d-0.35)*0.35; else if (d < 0.75) a = 0.13 + (d-0.55)*0.6; else a = 0.25 + (d-0.75)*2.5;
        }
        a *= opacity;
        a = Math.min(a, 1.0);
        const y = h/2 - a * (h/2) * 0.9;
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = '#8892b0';
    ctx.font = '9px sans-serif';
    ctx.fillText('0', 2, h-2);
    ctx.fillText('1', w-8, h-2);
}

function volPresetView(view) {
    if (!volCamera || !volControls) return;
    switch(view) {
        case 'front': volCamera.position.set(0, 0, 3.5); break;
        case 'side':  volCamera.position.set(3.5, 0, 0); break;
        case 'top':   volCamera.position.set(0, 3.5, 0); break;
        case 'iso':   volCamera.position.set(2.5, 2.5, 2.5); break;
    }
    volControls.target.set(0, 0, 0);
    volControls.update();
}

function resetView() {
    if (volCamera) { volCamera.position.set(2.8, 2.2, 2.8); volCamera.lookAt(0, 0, 0); }
    if (volControls) { volControls.target.set(0, 0, 0); volControls.update(); }
}

function toggleAnimation() {
    const btn = document.getElementById('play-btn');
    if (isPlaying) {
        isPlaying = false;
        btn.innerText = '▶ 播放';
        if (animTimer) clearInterval(animTimer);
    } else {
        isPlaying = true;
        btn.innerText = '⏸ 暂停';
        let ts = parseInt(document.getElementById('ctrl-timestep').value);
        if (ts >= 99) ts = 0;
        animTimer = setInterval(() => {
            document.getElementById('ctrl-timestep').value = ts;
            onTimestepDrag();
            ts++;
            if (ts > 99) ts = 0;
        }, 200);
    }
}

// ===== Initialize =====
window.addEventListener('DOMContentLoaded', () => {
    initVolume3D();
    preloadAllData().then(() => {
        hideLoadingScreen();
        onTimestepDrag();
    });
});

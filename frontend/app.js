const API_BASE = "/api";

let hourlyChart = null;
let pollutantChart = null;
let predictionChart = null;

const fallbackCurrent = {
    current_aqi: 137,
    aqi_label: "Unhealthy (Sensitive)",
    aqi_color: "#ffad3a",
    pm25: 52,
    pm10: 91,
    no2: 5.4,
    o3: 0.6,
    co: 0.5,
    temperature: 33,
    humidity: 54,
    wind_speed: 11,
    pressure: 1004,
    timestamp: new Date().toISOString()
};

const chartColors = {
    primary: "#57f1db",
    secondary: "#cebdff",
    tertiary: "#ffad3a",
    danger: "#ffb4ab",
    muted: "#bacac5",
    grid: "rgba(186, 202, 197, 0.12)"
};

async function fetchData(endpoint, fallback) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        if (payload && !payload.error) return payload;
    } catch (error) {
        console.warn(`Using fallback for ${endpoint}:`, error);
    }
    return fallback;
}

function formatNumber(value, decimals = 0) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "--";
    return number.toLocaleString("en-US", {
        maximumFractionDigits: decimals,
        minimumFractionDigits: decimals
    });
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function getAqiMeta(aqi) {
    const value = Number(aqi) || 0;
    if (value <= 50) return { label: "Good", tone: "success", icon: "check_circle", color: "#57f1db" };
    if (value <= 100) return { label: "Moderate", tone: "success", icon: "info", color: "#57f1db" };
    if (value <= 150) return { label: "Unhealthy (Sensitive)", tone: "warning", icon: "warning", color: "#ffad3a" };
    if (value <= 200) return { label: "Unhealthy", tone: "danger", icon: "report", color: "#ffb4ab" };
    if (value <= 300) return { label: "Very Unhealthy", tone: "danger", icon: "report", color: "#cebdff" };
    return { label: "Hazardous", tone: "danger", icon: "dangerous", color: "#ffb4ab" };
}

function pollutantStatus(value, type) {
    const thresholds = {
        pm25: [12, 35.4, 55.4],
        pm10: [54, 154, 254],
        no2: [40, 100, 200],
        o3: [60, 100, 180]
    }[type] || [40, 100, 180];

    const number = Number(value);
    if (!Number.isFinite(number)) return "--";
    if (number <= thresholds[0]) return "Excellent";
    if (number <= thresholds[1]) return "Good";
    if (number <= thresholds[2]) return "Moderate";
    return "High";
}

function updateLastUpdated(timestamp) {
    const date = timestamp ? new Date(timestamp) : new Date();
    document.getElementById("last-updated").textContent = `Last updated: ${date.toLocaleString()}`;
}

function renderAlert(aqi) {
    const meta = getAqiMeta(aqi);
    const alertSection = document.getElementById("alert-section");
    const alertTitle = document.getElementById("alert-title");
    const alertMessage = document.getElementById("alert-message");
    const alertIcon = document.getElementById("alert-icon");

    alertSection.className = `alert-card ${meta.tone}`;
    alertIcon.textContent = meta.icon;

    if (meta.tone === "danger") {
        alertTitle.textContent = "Health Alert";
        alertMessage.innerHTML = `AQI is <strong>${formatNumber(aqi)}</strong> - limit outdoor activity and use a mask if you go out.`;
    } else if (meta.tone === "warning") {
        alertTitle.textContent = "Attention Advised";
        alertMessage.innerHTML = `AQI is <strong>${formatNumber(aqi)}</strong> - sensitive groups should exercise caution outdoors.`;
    } else {
        alertTitle.textContent = "Breathing Easy";
        alertMessage.innerHTML = `AQI is <strong>${formatNumber(aqi)}</strong> - air quality is acceptable right now.`;
    }
}

function setMetric(id, value, unit, decimals = 0) {
    document.getElementById(id).textContent = `${formatNumber(value, decimals)} ${unit}`;
}

function setPollutant(type, value, max, decimals = 0) {
    document.getElementById(type).textContent = formatNumber(value, decimals);
    document.getElementById(`${type}-status`).textContent = pollutantStatus(value, type);
    document.getElementById(`${type}-bar`).style.width = `${clamp((Number(value) || 0) / max * 100, 4, 100)}%`;
}

function normalizeCurrent(data) {
    return {
        ...fallbackCurrent,
        ...(data || {}),
        current_aqi: Number(data?.current_aqi ?? data?.aqi ?? fallbackCurrent.current_aqi)
    };
}

function renderCurrentData(rawData) {
    const data = normalizeCurrent(rawData);
    const meta = getAqiMeta(data.current_aqi);

    document.getElementById("aqi-value").textContent = formatNumber(data.current_aqi);
    document.getElementById("aqi-label").textContent = data.aqi_label || meta.label;
    document.getElementById("aqi-label").style.color = data.aqi_color || meta.color;

    setPollutant("pm25", data.pm25, 180, 1);
    setPollutant("pm10", data.pm10, 220, 1);
    setPollutant("no2", data.no2, 120, 1);
    setPollutant("o3", data.o3, 160, 1);

    setMetric("temperature", data.temperature, "C", 1);
    setMetric("humidity", data.humidity, "%", 0);
    setMetric("wind-speed", data.wind_speed, "km/h", 1);
    setMetric("pressure", data.pressure, "hPa", 0);

    updateLastUpdated(data.timestamp);
    renderAlert(data.current_aqi);
}

function renderPrediction(data) {
    const payload = data || {
        predicted_pm25_24h: fallbackCurrent.pm25 + 6,
        predicted_aqi: fallbackCurrent.current_aqi + 12,
        predicted_label: "Unhealthy (Sensitive)"
    };

    document.getElementById("pred-pm25").textContent = `${formatNumber(payload.predicted_pm25_24h, 1)} ug/m3`;
    document.getElementById("pred-aqi").textContent = formatNumber(payload.predicted_aqi);
    document.getElementById("pred-label").textContent = payload.predicted_label || getAqiMeta(payload.predicted_aqi).label;
}

function makeFallbackForecast() {
    const now = new Date();
    return Array.from({ length: 72 }, (_, index) => {
        const hour = index + 1;
        const time = new Date(now.getTime() + hour * 60 * 60 * 1000);
        const wave = Math.sin(hour / 6) * 14;
        const daily = Math.sin(hour / 24 * Math.PI) * 10;
        const pm25 = Math.max(10, 48 + wave + daily + (index % 5));
        const pm10 = Math.max(18, 82 + wave * 1.1 + (index % 7));
        const no2 = Math.max(2, 18 + Math.cos(hour / 5) * 6);
        const o3 = Math.max(2, 42 + Math.sin(hour / 8) * 10);
        return {
            time: time.toISOString(),
            pm25: Number(pm25.toFixed(1)),
            pm10: Number(pm10.toFixed(1)),
            no2: Number(no2.toFixed(1)),
            o3: Number(o3.toFixed(1)),
            aqi: pm25ToAqi(pm25)
        };
    });
}

function pm25ToAqi(pm25) {
    const breakpoints = [
        [0, 12, 0, 50],
        [12.1, 35.4, 51, 100],
        [35.5, 55.4, 101, 150],
        [55.5, 150.4, 151, 200],
        [150.5, 250.4, 201, 300],
        [250.5, 350.4, 301, 400],
        [350.5, 500.4, 401, 500]
    ];
    const value = Math.max(0, Number(pm25) || 0);
    const match = breakpoints.find(([low, high]) => value >= low && value <= high);
    if (!match) return 500;
    const [cLow, cHigh, aLow, aHigh] = match;
    return Math.round(((aHigh - aLow) / (cHigh - cLow)) * (value - cLow) + aLow);
}

function makeDailyFromHourly(hourly) {
    const groups = new Map();
    hourly.forEach((item) => {
        const date = new Date(item.time);
        const key = date.toDateString();
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(item);
    });

    return Array.from(groups.entries()).slice(0, 3).map(([key, values]) => {
        const avgAqi = average(values.map((item) => item.aqi));
        const avgPm25 = average(values.map((item) => item.pm25));
        const meta = getAqiMeta(avgAqi);
        return {
            date: new Date(key).toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" }),
            aqi: Math.round(avgAqi),
            min_aqi: Math.round(Math.min(...values.map((item) => Number(item.aqi) || 0))),
            max_aqi: Math.round(Math.max(...values.map((item) => Number(item.aqi) || 0))),
            pm25: Number(avgPm25.toFixed(1)),
            label: meta.label
        };
    });
}

function average(values) {
    const valid = values.map(Number).filter(Number.isFinite);
    if (!valid.length) return 0;
    return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function renderDailyForecast(data, hourlyData) {
    const daily = Array.isArray(data) && data.length ? data : makeDailyFromHourly(hourlyData);
    const container = document.getElementById("daily-forecast");
    container.innerHTML = "";

    daily.slice(0, 3).forEach((item) => {
        const div = document.createElement("div");
        div.className = "daily-item";
        div.innerHTML = `
            <p class="date">${item.date}</p>
            <p class="aqi">${formatNumber(item.aqi)}</p>
            <span class="badge">${item.label || getAqiMeta(item.aqi).label}</span>
            <p class="range">AQI ${formatNumber(item.min_aqi)}-${formatNumber(item.max_aqi)}</p>
            <p class="pm25">PM2.5: ${formatNumber(item.pm25, 1)} ug/m3</p>
        `;
        container.appendChild(div);
    });
}

function chartDefaults() {
    Chart.defaults.color = chartColors.muted;
    Chart.defaults.font.family = "Inter, sans-serif";
    Chart.defaults.plugins.tooltip.backgroundColor = "rgba(10, 15, 13, 0.94)";
    Chart.defaults.plugins.tooltip.borderColor = "rgba(133, 148, 144, 0.24)";
    Chart.defaults.plugins.tooltip.borderWidth = 1;
}

function chartScales() {
    return {
        x: {
            grid: { color: "transparent" },
            ticks: { color: chartColors.muted, maxTicksLimit: 10 }
        },
        y: {
            beginAtZero: true,
            grid: { color: chartColors.grid },
            ticks: { color: chartColors.muted }
        }
    };
}

function renderHourlyChart(data) {
    const hourly = Array.isArray(data) && data.length ? data : makeFallbackForecast();
    const labels = hourly.map((item) => new Date(item.time).toLocaleString("en-US", { hour: "numeric", hour12: true }));

    if (hourlyChart) hourlyChart.destroy();
    hourlyChart = new Chart(document.getElementById("hourlyChart"), {
        type: "line",
        data: {
            labels,
            datasets: [{
                label: "AQI",
                data: hourly.map((item) => item.aqi),
                borderColor: chartColors.primary,
                backgroundColor: "rgba(87, 241, 219, 0.12)",
                borderWidth: 3,
                fill: true,
                tension: 0.35,
                pointRadius: 0,
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: chartScales()
        }
    });

    return hourly;
}

function renderPollutantChart(hourly) {
    const source = Array.isArray(hourly) && hourly.length ? hourly : makeFallbackForecast();
    const labels = source.slice(0, 24).map((item) => new Date(item.time).toLocaleString("en-US", { hour: "numeric", hour12: true }));

    if (pollutantChart) pollutantChart.destroy();
    pollutantChart = new Chart(document.getElementById("pollutantChart"), {
        type: "bar",
        data: {
            labels,
            datasets: [
                { label: "PM2.5", data: source.slice(0, 24).map((item) => item.pm25), backgroundColor: "rgba(87, 241, 219, 0.8)" },
                { label: "PM10", data: source.slice(0, 24).map((item) => item.pm10), backgroundColor: "rgba(206, 189, 255, 0.75)" },
                { label: "NO2", data: source.slice(0, 24).map((item) => item.no2), backgroundColor: "rgba(255, 173, 58, 0.72)" }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { boxWidth: 10, boxHeight: 10 } } },
            scales: chartScales()
        }
    });
}

function renderPredictionChart(hourly, prediction) {
    const source = (Array.isArray(hourly) && hourly.length ? hourly : makeFallbackForecast()).slice(0, 24);
    const predictedAqi = Number(prediction?.predicted_aqi);
    const labels = source.map((item) => new Date(item.time).toLocaleString("en-US", { hour: "numeric", hour12: true }));

    if (predictionChart) predictionChart.destroy();
    predictionChart = new Chart(document.getElementById("predictionChart"), {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "Forecast AQI",
                    data: source.map((item) => item.aqi),
                    borderColor: chartColors.secondary,
                    backgroundColor: "rgba(206, 189, 255, 0.1)",
                    borderWidth: 2,
                    tension: 0.35,
                    pointRadius: 0,
                    fill: true
                },
                {
                    label: "Model 24h AQI",
                    data: source.map((_, index) => index === source.length - 1 ? predictedAqi || source[index].aqi : null),
                    borderColor: chartColors.tertiary,
                    backgroundColor: chartColors.tertiary,
                    pointRadius: 6,
                    pointHoverRadius: 7,
                    showLine: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { boxWidth: 10, boxHeight: 10 } } },
            scales: chartScales()
        }
    });
}

function setupTabs() {
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabButtons.forEach((button) => {
        button.addEventListener("click", () => {
            tabButtons.forEach((btn) => btn.classList.remove("active"));
            tabContents.forEach((content) => content.classList.remove("active"));
            button.classList.add("active");
            document.getElementById(button.dataset.tab).classList.add("active");
            requestAnimationFrame(() => {
                hourlyChart?.resize();
                pollutantChart?.resize();
                predictionChart?.resize();
            });
        });
    });
}

async function loadDashboard() {
    const fallbackHourly = makeFallbackForecast();
    const [currentData, predictionData, hourlyData, dailyData] = await Promise.all([
        fetchData("/current", fallbackCurrent),
        fetchData("/predict", null),
        fetchData("/forecast", fallbackHourly),
        fetchData("/daily-forecast", null)
    ]);

    renderCurrentData(currentData);
    renderPrediction(predictionData);
    const hourly = renderHourlyChart(hourlyData);
    renderPollutantChart(hourly);
    renderPredictionChart(hourly, predictionData);
    renderDailyForecast(dailyData, hourly);
}

function init() {
    chartDefaults();
    setupTabs();
    loadDashboard();
    setInterval(loadDashboard, 300000);
}

document.addEventListener("DOMContentLoaded", init);

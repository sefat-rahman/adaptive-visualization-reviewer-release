'use strict';

const APP_CONFIG = window.APP_CONFIG || {};
const STATE_META = APP_CONFIG.stateMeta || {};
const FPS_THRESHOLD_COVERAGE = APP_CONFIG.fpsThresholdCoverage || {};
const STATIC_SNAPSHOT_COVERAGE = APP_CONFIG.staticSnapshotCoverage || null;
const STATIC_DEMO_MODE = Boolean(APP_CONFIG.staticDemo || STATIC_SNAPSHOT_COVERAGE);
const STATE_ROWS = STATE_META.rows || [];
const STATE_FIPS_TO_CODE = STATE_META.fipsToCode || {};
const STATE_CODE_TO_NAME = STATE_META.codeToName || {};
const STATE_CODE_TO_FIPS = Object.fromEntries(STATE_ROWS.map((row) => [row.code, row.fips]));
const AVAILABLE_STATES = new Set(STATE_ROWS.map((row) => row.code));
const RETAIN_PERCENTAGES = [1, 3, 4, 5, 10, 15, 20, 25, 26, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100];
const COUNTRY_DISTANCE_BASELINE = 'country_1pct_distance';
const COUNTY_10_DISTANCE_BASELINE = 'county_10pct_distance';
const COUNTY_20_DISTANCE_BASELINE = 'county_20pct_distance';
const DEFAULT_TOPOLOGY_BASELINE = COUNTY_20_DISTANCE_BASELINE;
const DISTANCE_BASELINE_LABELS = {
  [COUNTRY_DISTANCE_BASELINE]: 'Country 1% distance',
  [COUNTY_10_DISTANCE_BASELINE]: 'County 10% distance',
  [COUNTY_20_DISTANCE_BASELINE]: 'County 20% distance',
};
const DISTANCE_TOPOLOGY_BASELINE_OPTIONS = [
  COUNTRY_DISTANCE_BASELINE,
  COUNTY_10_DISTANCE_BASELINE,
  COUNTY_20_DISTANCE_BASELINE,
];
const COUNTRY_TOPOLOGY_BASELINE_PERCENTAGES = DISTANCE_TOPOLOGY_BASELINE_OPTIONS;
const STATE_TOPOLOGY_BASELINE_PERCENTAGES = DISTANCE_TOPOLOGY_BASELINE_OPTIONS;
const COUNTY_TOPOLOGY_BASELINE_PERCENTAGES = DISTANCE_TOPOLOGY_BASELINE_OPTIONS;

const COUNTRY_EXTENT = ol.proj.transformExtent(
  [-125, 24, -66, 49.8],
  'EPSG:4326',
  'EPSG:3857'
);
const COUNTRY_FIT_PADDING = [36, 34, 36, 34];
const STATE_FIT_PADDING = [32, 32, 32, 32];
const COUNTY_FIT_PADDING = [34, 34, 34, 34];
const POINT_FILL_COLORS = {
  default: 'rgba(21, 101, 192, 0.78)',
  pixel: 'rgba(111, 104, 160, 0.80)',
  fps_threshold: 'rgba(46, 125, 50, 0.82)',
  random: 'rgba(154, 74, 49, 0.80)',
};
const PERSISTENCE_DIAGRAM_COLOR = 'rgba(76, 112, 147, 0.78)';
const PERSISTENCE_DIAGONAL_COLOR = 'rgba(118, 126, 135, 0.72)';
const REGION_COLORS = {
  default: {
    fill: 'rgba(214, 226, 255, 0.72)',
    stroke: '#1565c0',
    innerStroke: 'rgba(79, 129, 189, 0.75)',
    hoverFill: 'rgba(255, 255, 255, 0.06)',
    hoverStroke: '#0d47a1',
    precomputedFill: 'rgba(21, 101, 192, 0.13)',
    precomputedCountyFill: 'rgba(21, 101, 192, 0.18)',
    precomputedStroke: '#0d47a1',
    buttonBg: 'rgba(21, 101, 192, 0.14)',
    buttonHoverBg: 'rgba(21, 101, 192, 0.22)',
    buttonBorder: 'rgba(21, 101, 192, 0.42)',
    buttonText: '#0d47a1',
  },
  pixel: {
    fill: 'rgba(226, 224, 239, 0.64)',
    stroke: '#6f68a0',
    innerStroke: 'rgba(111, 104, 160, 0.56)',
    hoverFill: 'rgba(111, 104, 160, 0.10)',
    hoverStroke: '#54507c',
    precomputedFill: 'rgba(111, 104, 160, 0.16)',
    precomputedCountyFill: 'rgba(111, 104, 160, 0.22)',
    precomputedStroke: '#48436e',
    buttonBg: 'rgba(111, 104, 160, 0.16)',
    buttonHoverBg: 'rgba(111, 104, 160, 0.25)',
    buttonBorder: 'rgba(111, 104, 160, 0.46)',
    buttonText: '#48436e',
  },
  fps_threshold: {
    fill: 'rgba(210, 234, 214, 0.66)',
    stroke: '#2e7d32',
    innerStroke: 'rgba(46, 125, 50, 0.58)',
    hoverFill: 'rgba(46, 125, 50, 0.10)',
    hoverStroke: '#1b5e20',
    precomputedFill: 'rgba(46, 125, 50, 0.15)',
    precomputedCountyFill: 'rgba(46, 125, 50, 0.21)',
    precomputedStroke: '#1b5e20',
    buttonBg: 'rgba(46, 125, 50, 0.16)',
    buttonHoverBg: 'rgba(46, 125, 50, 0.25)',
    buttonBorder: 'rgba(46, 125, 50, 0.44)',
    buttonText: '#1b5e20',
  },
  random: {
    fill: 'rgba(238, 221, 214, 0.62)',
    stroke: '#9a4a31',
    innerStroke: 'rgba(154, 74, 49, 0.58)',
    hoverFill: 'rgba(154, 74, 49, 0.10)',
    hoverStroke: '#743724',
    precomputedFill: 'rgba(154, 74, 49, 0.15)',
    precomputedCountyFill: 'rgba(154, 74, 49, 0.22)',
    precomputedStroke: '#743724',
    buttonBg: 'rgba(154, 74, 49, 0.16)',
    buttonHoverBg: 'rgba(154, 74, 49, 0.25)',
    buttonBorder: 'rgba(154, 74, 49, 0.44)',
    buttonText: '#743724',
  },
};
const METHOD_LABELS = {
  all: 'All Points',
  pixel: 'Pixel Occupancy',
  random: 'Random',
  fps_threshold: 'FPS threshold',
};
const ANALYSIS_LABELS = {
  topological: 'Structural',
  statistical: 'Statistical',
  density: 'Distributional',
};

const appState = {
  zoom: 'country',
  state: null,
  county: null,
  method: 'pixel',
  analysisProperty: 'topological',
  retainPercentage: 50,
  errorThreshold: 0.5,
  topologyBaselinePercentage: DEFAULT_TOPOLOGY_BASELINE,
  showPrecomputedHighlights: true,
};

const sources = {
  nation: new ol.source.Vector(),
  states: new ol.source.Vector(),
  stateLabels: new ol.source.Vector(),
  selectedState: new ol.source.Vector(),
  stateCountyBoundaries: new ol.source.Vector(),
  hoveredCountyBoundary: new ol.source.Vector(),
  countyLabels: new ol.source.Vector(),
  countyBoundary: new ol.source.Vector(),
  points: new ol.source.Vector(),
};

const layersRef = {};
const pointStyleCache = {};
const regionStyleCache = {};

let map = null;
let countyHoverTooltip = null;
let pixelViewportRefreshTimer = null;
let fpsGenerationPollTimer = null;
let lastFpsThresholdStatus = null;
let stateTopologyLoaded = false;
let countyTopologyLoaded = false;
const countyFeaturesByState = new Map();
let analysisRequestToken = 0;
let pointsRequestToken = 0;
let currentPageMode = {
  capture: false,
  mapOnly: false,
  hideTopbar: false,
};

function setStartupError(message, error = null) {
  if (error) {
    console.error(error);
  }

  const mapStatus = document.getElementById('map-status');
  if (mapStatus) {
    mapStatus.textContent = message;
  }

  const analysisTitle = document.getElementById('analysis-title');
  const analysisSubtitle = document.getElementById('analysis-subtitle');
  const analysisContent = document.getElementById('analysis-content');
  if (analysisTitle) {
    analysisTitle.textContent = 'Startup Error';
  }
  if (analysisSubtitle) {
    analysisSubtitle.textContent = 'The map viewer could not finish loading.';
  }
  if (analysisContent) {
    analysisContent.innerHTML = `<div class="analysis-placeholder">${message}</div>`;
  }
}

function getControlDefaultsFactory() {
  if (!window.ol || !window.ol.control) {
    return null;
  }

  if (typeof window.ol.control.defaults === 'function') {
    return window.ol.control.defaults;
  }

  if (window.ol.control.defaults && typeof window.ol.control.defaults.defaults === 'function') {
    return window.ol.control.defaults.defaults;
  }

  return null;
}

function topologyBaselineOptionsForZoom(zoom) {
  if (zoom === 'country') {
    return COUNTRY_TOPOLOGY_BASELINE_PERCENTAGES;
  }
  if (zoom === 'county') {
    return COUNTY_TOPOLOGY_BASELINE_PERCENTAGES;
  }
  return STATE_TOPOLOGY_BASELINE_PERCENTAGES;
}

function coerceTopologyBaselinePercentage(value, zoom = appState.zoom) {
  const options = topologyBaselineOptionsForZoom(zoom);
  const stringValue = String(value || '').trim();
  if (DISTANCE_BASELINE_LABELS[stringValue] && options.includes(stringValue)) {
    return stringValue;
  }
  const numericValue = Number(value);
  if (Number.isFinite(numericValue) && options.includes(numericValue)) {
    return numericValue;
  }
  return options.includes(DEFAULT_TOPOLOGY_BASELINE) ? DEFAULT_TOPOLOGY_BASELINE : options[0];
}

function topologyBaselineLabel(value) {
  if (DISTANCE_BASELINE_LABELS[value]) {
    return DISTANCE_BASELINE_LABELS[value];
  }
  return `${value}%`;
}

function analysisLabel(value = appState.analysisProperty) {
  return ANALYSIS_LABELS[value] || value;
}

function methodCoverage(method = appState.method) {
  const staticMethods = STATIC_SNAPSHOT_COVERAGE?.methods || {};
  if (staticMethods[method]) {
    return staticMethods[method];
  }
  if (method === 'fps_threshold') {
    return FPS_THRESHOLD_COVERAGE;
  }
  return { states: [], counties: {} };
}

function methodHasPrecomputedCoverage(method = appState.method) {
  const coverage = methodCoverage(method);
  if (method === 'random') {
    const retainPercentages = coverage.retainPercentages || [];
    if (retainPercentages.length && !retainPercentages.includes(Number(appState.retainPercentage))) {
      return false;
    }
  }
  const hasStates = Array.isArray(coverage.states) && coverage.states.length > 0;
  const hasCounties = Object.values(coverage.counties || {})
    .some((counties) => Array.isArray(counties) && counties.length > 0);
  return hasStates || hasCounties;
}

function isStatePrecomputedForMethod(stateCode, method = appState.method) {
  if (!stateCode) {
    return false;
  }
  return (methodCoverage(method).states || []).includes(stateCode);
}

function isCountyPrecomputedForMethod(stateCode, countyName, method = appState.method) {
  if (!stateCode || !countyName) {
    return false;
  }
  return ((methodCoverage(method).counties || {})[stateCode] || []).includes(countyName);
}

function randomPrecomputeNote() {
  if (!STATIC_DEMO_MODE || appState.method !== 'random') {
    return '';
  }
  const percentages = methodCoverage('random').retainPercentages || [];
  if (!percentages.length) {
    return '';
  }
  const label = percentages.map((percentage) => `${percentage}%`).join(', ');
  return ` Precomputed random retains: ${label}.`;
}

function syncTopologyBaselineOptions(zoom = appState.zoom) {
  const select = document.getElementById('topology-baseline-select');
  if (!select) {
    return;
  }

  const options = topologyBaselineOptionsForZoom(zoom);
  if (!options.includes(appState.topologyBaselinePercentage)) {
    appState.topologyBaselinePercentage = coerceTopologyBaselinePercentage(
      appState.topologyBaselinePercentage,
      zoom
    );
  }

  select.replaceChildren(
    ...options.map((percentage) => {
      const option = document.createElement('option');
      option.value = String(percentage);
      option.textContent = topologyBaselineLabel(percentage);
      return option;
    })
  );
  select.value = String(appState.topologyBaselinePercentage);
}

function parseUrlConfig() {
  const params = new URLSearchParams(window.location.search);
  const zoom = params.get('zoom');
  const method = params.get('method');
  const analysis = params.get('analysis');
  const retain = Number(params.get('retain'));
  const threshold = Number(params.get('threshold'));
  const baseline = params.get('baseline');
  const state = params.get('state');
  const county = params.get('county');
  const capture = params.get('capture');
  const layout = params.get('layout');
  const chromeMode = params.get('chrome');
  const normalizedZoom = ['country', 'state', 'county'].includes(zoom) ? zoom : 'country';

  return {
    zoom: normalizedZoom,
    method: ['pixel', 'all', 'fps_threshold', 'random'].includes(method) ? method : appState.method,
    analysisProperty: ['statistical', 'density', 'topological'].includes(analysis) ? analysis : appState.analysisProperty,
    retainPercentage: RETAIN_PERCENTAGES.includes(retain) ? retain : appState.retainPercentage,
    errorThreshold: [0.05, 0.1, 0.25, 0.5, 0.75, 1.0].includes(threshold) ? threshold : appState.errorThreshold,
    topologyBaselinePercentage: coerceTopologyBaselinePercentage(baseline, normalizedZoom),
    state: state && AVAILABLE_STATES.has(state) ? state : null,
    county: county ? county : null,
    capture: capture === '1',
    mapOnly: layout === 'map-only',
    hideTopbar: chromeMode === '0',
  };
}

function applyPageMode(config) {
  currentPageMode = {
    capture: Boolean(config.capture),
    mapOnly: Boolean(config.mapOnly),
    hideTopbar: Boolean(config.hideTopbar),
  };

  document.body.classList.toggle('capture-mode', currentPageMode.capture);
  document.body.classList.toggle('capture-map-only', currentPageMode.capture && currentPageMode.mapOnly);
  document.body.classList.toggle('capture-hide-topbar', currentPageMode.capture && currentPageMode.hideTopbar);
}

function applyInitialConfig(config) {
  appState.method = config.method;
  appState.analysisProperty = config.analysisProperty;
  appState.retainPercentage = config.retainPercentage;
  appState.errorThreshold = config.errorThreshold;
  appState.topologyBaselinePercentage = coerceTopologyBaselinePercentage(
    config.topologyBaselinePercentage,
    config.zoom
  );

  const methodSelect = document.getElementById('method-select');
  const retainSelect = document.getElementById('retain-select');
  const analysisSelect = document.getElementById('analysis-select');
  const thresholdSelect = document.getElementById('threshold-select');
  const topologyBaselineSelect = document.getElementById('topology-baseline-select');

  if (methodSelect) {
    methodSelect.value = appState.method;
  }
  if (retainSelect) {
    retainSelect.value = String(appState.retainPercentage);
  }
  if (analysisSelect) {
    analysisSelect.value = appState.analysisProperty;
  }
  if (thresholdSelect) {
    thresholdSelect.value = String(appState.errorThreshold.toFixed(2));
  }
  syncTopologyBaselineOptions(config.zoom);
  if (topologyBaselineSelect) {
    topologyBaselineSelect.value = String(appState.topologyBaselinePercentage);
  }
}

function normalizedInitialRoute(config) {
  if (config.zoom === 'county' && config.state && config.county) {
    return { zoom: 'county', state: config.state, county: config.county };
  }
  if ((config.zoom === 'state' || config.zoom === 'county') && config.state) {
    return { zoom: 'state', state: config.state, county: null };
  }
  return { zoom: 'country', state: null, county: null };
}

function syncUrlToState() {
  const params = new URLSearchParams();
  params.set('zoom', appState.zoom);
  params.set('method', appState.method);
  params.set('analysis', appState.analysisProperty);
  params.set('retain', String(appState.retainPercentage));
  params.set('threshold', appState.errorThreshold.toFixed(2));
  params.set('baseline', String(appState.topologyBaselinePercentage));
  if (appState.state) {
    params.set('state', appState.state);
  }
  if (appState.county) {
    params.set('county', appState.county);
  }
  if (currentPageMode.capture) {
    params.set('capture', '1');
  }
  if (currentPageMode.mapOnly) {
    params.set('layout', 'map-only');
  }
  if (currentPageMode.hideTopbar) {
    params.set('chrome', '0');
  }

  const nextUrl = `${window.location.pathname}?${params.toString()}`;
  window.history.replaceState(null, '', nextUrl);
}

function initApp() {
  try {
    if (!window.ol || !window.ol.Map || !window.ol.View) {
      throw new Error('OpenLayers failed to load.');
    }
    if (!window.Plotly) {
      console.warn('Plotly failed to load. Analysis charts may be unavailable.');
    }

    const initialConfig = parseUrlConfig();
    applyPageMode(initialConfig);
    applyInitialConfig(initialConfig);

    const controlDefaultsFactory = getControlDefaultsFactory();
    if (!controlDefaultsFactory) {
      throw new Error('OpenLayers controls factory is unavailable.');
    }

    map = new ol.Map({
      target: 'map',
      layers: buildLayers(),
      view: new ol.View({
        center: ol.proj.fromLonLat([-98.5, 39.8]),
        zoom: 4.15,
        minZoom: 3.4,
        maxZoom: 18,
        extent: COUNTRY_EXTENT,
      }),
      controls: controlDefaultsFactory({
        attribution: false,
        rotate: false,
        zoom: false,
      }).extend([
        new ol.control.ScaleLine(),
      ]),
    });

    map.on('singleclick', handleMapClick);
    map.on('pointermove', handlePointerMove);
    map.on('change:size', schedulePixelViewportRefresh);

    countyHoverTooltip = document.createElement('div');
    countyHoverTooltip.className = 'county-hover-tooltip';
    countyHoverTooltip.hidden = true;
    map.getTargetElement().appendChild(countyHoverTooltip);
    map.getViewport().addEventListener('mouseleave', clearCountyHover);

    document.getElementById('method-select').addEventListener('change', (event) => {
      appState.method = event.target.value;
      updateRetainControl();
      refreshMethodStyles();
      navigate(appState.zoom, appState.state, appState.county);
    });

    document.getElementById('threshold-select').addEventListener('change', (event) => {
      appState.errorThreshold = Number(event.target.value || 0.5);
      navigate(appState.zoom, appState.state, appState.county);
    });

    document.getElementById('topology-baseline-select').addEventListener('change', (event) => {
      appState.topologyBaselinePercentage = coerceTopologyBaselinePercentage(event.target.value, appState.zoom);
      navigate(appState.zoom, appState.state, appState.county);
    });

    document.getElementById('retain-select').addEventListener('change', (event) => {
      appState.retainPercentage = Number(event.target.value || 50);
      navigate(appState.zoom, appState.state, appState.county);
    });

    document.getElementById('analysis-select').addEventListener('change', (event) => {
      appState.analysisProperty = event.target.value;
      syncUrlToState();
      loadPoints().then((pointsReady) => {
        if (pointsReady !== false) {
          loadAnalysis();
        }
      });
    });

    document.getElementById('back-btn').addEventListener('click', () => {
      if (appState.zoom === 'county') {
        navigate('state', appState.state, null);
      } else if (appState.zoom === 'state') {
        navigate('country');
      }
    });

    document.getElementById('precompute-toggle-btn').addEventListener('click', () => {
      appState.showPrecomputedHighlights = !appState.showPrecomputedHighlights;
      updatePrecomputeToggle();
      refreshMethodStyles();
    });

    updateRetainControl();

    loadCountryBoundaries()
      .then(() => {
        const initialRoute = normalizedInitialRoute(initialConfig);
        return navigate(initialRoute.zoom, initialRoute.state, initialRoute.county);
      })
      .catch((error) => {
        setStartupError('Could not load U.S. boundary data. Check the cached topology files and refresh the page.', error);
      });
  } catch (error) {
    setStartupError('Map initialization failed. Please hard refresh the page and try again.', error);
  }
}

function buildLayers() {
  const nationLayer = new ol.layer.Vector({
    source: sources.nation,
    style: new ol.style.Style({
      fill: new ol.style.Fill({ color: 'rgba(232, 240, 246, 0.95)' }),
      stroke: new ol.style.Stroke({ color: '#90a4ae', width: 1.3 }),
    }),
    zIndex: 1,
  });

  const statesLayer = new ol.layer.Vector({
    source: sources.states,
    style: (feature) => getStateBoundaryStyle(feature),
    zIndex: 2,
  });

  const stateLabelsLayer = new ol.layer.Vector({
    source: sources.stateLabels,
    declutter: true,
    style: (feature) => getStateLabelStyle(feature),
    zIndex: 8,
  });

  const selectedStateLayer = new ol.layer.Vector({
    source: sources.selectedState,
    style: () => getRegionStyle('selected', appState.method),
    zIndex: 4,
  });

  const stateCountyBoundariesLayer = new ol.layer.Vector({
    source: sources.stateCountyBoundaries,
    style: (feature) => getCountyBoundaryStyle(feature),
    zIndex: 4.5,
  });

  const hoveredCountyBoundaryLayer = new ol.layer.Vector({
    source: sources.hoveredCountyBoundary,
    style: () => getRegionStyle('hovered-boundary', appState.method),
    zIndex: 5.2,
  });

  const countyLabelsLayer = new ol.layer.Vector({
    source: sources.countyLabels,
    declutter: true,
    style: (feature) => createLabelBadgeStyle(feature.get('label')),
    zIndex: 9,
  });

  const countyBoundaryLayer = new ol.layer.Vector({
    source: sources.countyBoundary,
    style: () => getRegionStyle('selected', appState.method),
    zIndex: 6,
  });

  const pointsLayer = new ol.layer.VectorImage({
    source: sources.points,
    style: () => getPointStyle(appState.zoom, appState.method),
    zIndex: 7,
  });

  layersRef.nation = nationLayer;
  layersRef.states = statesLayer;
  layersRef.stateLabels = stateLabelsLayer;
  layersRef.selectedState = selectedStateLayer;
  layersRef.stateCountyBoundaries = stateCountyBoundariesLayer;
  layersRef.hoveredCountyBoundary = hoveredCountyBoundaryLayer;
  layersRef.countyLabels = countyLabelsLayer;
  layersRef.countyBoundary = countyBoundaryLayer;
  layersRef.points = pointsLayer;

  return [
    nationLayer,
    statesLayer,
    stateLabelsLayer,
    selectedStateLayer,
    stateCountyBoundariesLayer,
    hoveredCountyBoundaryLayer,
    countyLabelsLayer,
    countyBoundaryLayer,
    pointsLayer,
  ];
}

function getPointStyle(zoom, method) {
  const styleKey = `${zoom}:${method}`;
  if (pointStyleCache[styleKey]) {
    return pointStyleCache[styleKey];
  }

  const radius = zoom === 'country' ? 3.2 : 4.2;
  const strokeWidth = 0.6;
  const fillColor = POINT_FILL_COLORS[method] || POINT_FILL_COLORS.default;
  pointStyleCache[styleKey] = new ol.style.Style({
    image: new ol.style.Circle({
      radius,
      fill: new ol.style.Fill({ color: fillColor }),
      stroke: new ol.style.Stroke({
        color: 'rgba(255,255,255,0.94)',
        width: strokeWidth,
      }),
    }),
  });
  return pointStyleCache[styleKey];
}

function getStateBoundaryStyle(feature) {
  return showPrecomputeHighlights() && isStatePrecomputedForMethod(feature.get('stateCode'))
    ? getRegionStyle('precomputed-state', appState.method)
    : getRegionStyle('state-boundary', appState.method);
}

function getStateLabelStyle(feature) {
  const isPrecomputed = showPrecomputeHighlights()
    && isStatePrecomputedForMethod(feature.get('stateCode'));
  const colors = REGION_COLORS[appState.method] || REGION_COLORS.default;
  return createTextStyle(feature.get('label'), {
    font: `${isPrecomputed ? '800' : '600'} 11px "Segoe UI", sans-serif`,
    fill: isPrecomputed ? colors.hoverStroke : '#243b53',
    stroke: '#ffffff',
    strokeWidth: isPrecomputed ? 5.2 : 4.5,
  });
}

function getCountyBoundaryStyle(feature) {
  return showPrecomputeHighlights()
    && isCountyPrecomputedForMethod(appState.state, feature.get('county'))
    ? getRegionStyle('precomputed-county', appState.method)
    : getRegionStyle('inner-boundary', appState.method);
}

function showPrecomputeHighlights() {
  return methodHasPrecomputedCoverage(appState.method)
    && appState.showPrecomputedHighlights
    && ['country', 'state'].includes(appState.zoom);
}

function getRegionStyle(role, method) {
  const styleKey = `${role}:${method}`;
  if (regionStyleCache[styleKey]) {
    return regionStyleCache[styleKey];
  }

  const colors = REGION_COLORS[method] || REGION_COLORS.default;
  if (role === 'state-boundary') {
    regionStyleCache[styleKey] = new ol.style.Style({
      fill: new ol.style.Fill({ color: 'rgba(250, 252, 253, 0.96)' }),
      stroke: new ol.style.Stroke({ color: '#627d98', width: 1.05 }),
    });
    return regionStyleCache[styleKey];
  }

  if (role === 'precomputed-state') {
    regionStyleCache[styleKey] = new ol.style.Style({
      fill: new ol.style.Fill({ color: colors.precomputedFill || colors.hoverFill }),
      stroke: new ol.style.Stroke({ color: colors.precomputedStroke || colors.hoverStroke, width: 3.15 }),
    });
    return regionStyleCache[styleKey];
  }

  if (role === 'precomputed-county') {
    regionStyleCache[styleKey] = new ol.style.Style({
      fill: new ol.style.Fill({ color: colors.precomputedCountyFill || colors.hoverFill }),
      stroke: new ol.style.Stroke({ color: colors.precomputedStroke || colors.hoverStroke, width: 3 }),
    });
    return regionStyleCache[styleKey];
  }

  if (role === 'inner-boundary') {
    regionStyleCache[styleKey] = new ol.style.Style({
      fill: new ol.style.Fill({ color: 'rgba(0, 0, 0, 0)' }),
      stroke: new ol.style.Stroke({ color: colors.innerStroke, width: 0.9 }),
    });
    return regionStyleCache[styleKey];
  }

  if (role === 'hovered-boundary') {
    regionStyleCache[styleKey] = new ol.style.Style({
      fill: new ol.style.Fill({ color: colors.hoverFill }),
      stroke: new ol.style.Stroke({ color: colors.hoverStroke, width: 2.1 }),
    });
    return regionStyleCache[styleKey];
  }

  regionStyleCache[styleKey] = new ol.style.Style({
    fill: new ol.style.Fill({ color: colors.fill }),
    stroke: new ol.style.Stroke({ color: colors.stroke, width: 2.3 }),
  });
  return regionStyleCache[styleKey];
}

function refreshMethodStyles() {
  ['states', 'stateLabels', 'selectedState', 'stateCountyBoundaries', 'hoveredCountyBoundary', 'countyBoundary', 'points']
    .forEach((key) => {
      if (layersRef[key]) {
        layersRef[key].changed();
      }
    });
}

function syncLayerVisibility() {
  const isCountry = appState.zoom === 'country';
  const isState = appState.zoom === 'state';
  const isCounty = appState.zoom === 'county';

  if (layersRef.nation) layersRef.nation.setVisible(isCountry);
  if (layersRef.states) layersRef.states.setVisible(isCountry);
  if (layersRef.stateLabels) layersRef.stateLabels.setVisible(isCountry);
  if (layersRef.selectedState) layersRef.selectedState.setVisible(isState);
  if (layersRef.stateCountyBoundaries) layersRef.stateCountyBoundaries.setVisible(isState);
  if (layersRef.hoveredCountyBoundary) layersRef.hoveredCountyBoundary.setVisible(isState);
  if (layersRef.countyLabels) layersRef.countyLabels.setVisible(false);
  if (layersRef.countyBoundary) layersRef.countyBoundary.setVisible(isCounty);
  if (layersRef.points) layersRef.points.setVisible(true);
}
async function loadCountryBoundaries() {
  if (stateTopologyLoaded) {
    return;
  }

  const [nationResp, statesResp] = await Promise.all([
    fetch('/api/boundaries/us/nation'),
    fetch('/api/boundaries/us/states'),
  ]);
  const nationTopo = await nationResp.json();
  const statesTopo = await statesResp.json();

  const topoFormat = new ol.format.TopoJSON();

  const nationFeatures = topoFormat.readFeatures(nationTopo, {
    featureProjection: 'EPSG:3857',
  });
  const stateFeatures = topoFormat.readFeatures(statesTopo, {
    featureProjection: 'EPSG:3857',
  })
    .filter((feature) => {
      const code = codeFromFeature(feature);
      return Boolean(code && AVAILABLE_STATES.has(code));
    });

  stateFeatures.forEach((feature) => {
    const code = codeFromFeature(feature);
    feature.set('stateCode', code);
    feature.set('stateName', STATE_CODE_TO_NAME[code] || code);
    feature.set('kind', 'state-boundary');
  });

  sources.nation.clear();
  sources.states.clear();
  sources.stateLabels.clear();
  sources.nation.addFeatures(nationFeatures);
  sources.states.addFeatures(stateFeatures);
  sources.stateLabels.addFeatures(
    stateFeatures.map((feature) => makeStateLabelFeature(feature))
  );
  stateTopologyLoaded = true;
}

async function ensureCountyBoundaryTopology() {
  if (countyTopologyLoaded) {
    return;
  }

  const response = await fetch('/api/boundaries/us/counties');
  const countiesTopo = await response.json();
  const topoFormat = new ol.format.TopoJSON();
  const countyFeatures = topoFormat.readFeatures(countiesTopo, {
    featureProjection: 'EPSG:3857',
  });

  countyFeatures.forEach((feature) => {
    const countyFips = countyFipsFromFeature(feature);
    if (!countyFips) {
      return;
    }
    const stateFips = countyFips.slice(0, 2);
    const bucket = countyFeaturesByState.get(stateFips) || [];
    feature.set('countyFips', countyFips);
    feature.set('stateFips', stateFips);
    feature.set('county', feature.get('name') || (feature.getProperties() || {}).name || '');
    feature.set('kind', 'state-county-boundary');
    bucket.push(feature);
    countyFeaturesByState.set(stateFips, bucket);
  });

  countyTopologyLoaded = true;
}

function prepareStateCountyBoundaries() {
  sources.stateCountyBoundaries.clear();
  const stateFips = STATE_CODE_TO_FIPS[appState.state];
  if (!stateFips) {
    return;
  }

  const features = countyFeaturesByState.get(stateFips) || [];
  sources.stateCountyBoundaries.addFeatures(features.map((feature) => {
    const clone = feature.clone();
    return clone;
  }));
}

async function navigate(zoom, state = null, county = null) {
  stopFpsGenerationPolling();
  appState.zoom = zoom;
  appState.state = state;
  appState.county = county;
  updateBackButton();
  updateRetainControl();
  updateBreadcrumb();
  syncUrlToState();
  setStatus('Loading data...');

  clearSourcesForView();
  syncLayerVisibility();
  refreshMethodStyles();

  if (zoom === 'country') {
    map.getView().fit(COUNTRY_EXTENT, {
      padding: COUNTRY_FIT_PADDING,
      duration: 250,
      maxZoom: 5,
    });
  } else if (zoom === 'state') {
    prepareSelectedState();
    await ensureCountyBoundaryTopology();
    prepareStateCountyBoundaries();
    fitSelectedState();
  } else if (zoom === 'county') {
    await loadCountyBoundary();
    fitCountyBoundary();
  }

  const pointsReady = await loadPoints();
  if (pointsReady !== false) {
    loadAnalysis();
  }
}

function clearSourcesForView() {
  sources.selectedState.clear();
  sources.stateCountyBoundaries.clear();
  sources.hoveredCountyBoundary.clear();
  sources.countyLabels.clear();
  sources.countyBoundary.clear();
  sources.points.clear();
  clearCountyHover();
}

function updateBackButton() {
  document.getElementById('back-btn').disabled = appState.zoom === 'country';
}

function updateRetainControl() {
  const retainSelect = document.getElementById('retain-select');
  const thresholdSelect = document.getElementById('threshold-select');
  const topologyBaselineSelect = document.getElementById('topology-baseline-select');
  if (retainSelect) {
    retainSelect.value = String(appState.retainPercentage);
    const retainEnabled = appState.method === 'random';
    retainSelect.disabled = !retainEnabled;
  }
  if (thresholdSelect) {
    thresholdSelect.value = appState.errorThreshold.toFixed(2);
    thresholdSelect.disabled = appState.method !== 'fps_threshold';
  }
  if (topologyBaselineSelect) {
    syncTopologyBaselineOptions();
    topologyBaselineSelect.disabled = appState.method !== 'fps_threshold';
  }
  updatePrecomputeToggle();
}

function updatePrecomputeToggle() {
  const button = document.getElementById('precompute-toggle-btn');
  if (!button) {
    return;
  }

  const visible = methodHasPrecomputedCoverage(appState.method)
    && ['country', 'state'].includes(appState.zoom);
  button.hidden = !visible;
  if (!visible) {
    return;
  }

  const scopeLabel = appState.zoom === 'country' ? 'states' : 'counties';
  const actionLabel = appState.showPrecomputedHighlights ? 'Hide' : 'Show';
  const methodLabel = METHOD_LABELS[appState.method] || appState.method;
  const colors = REGION_COLORS[appState.method] || REGION_COLORS.default;
  button.textContent = `${actionLabel} precomputed ${scopeLabel}`;
  button.title = `${methodLabel} precomputed ${scopeLabel} bundled in this viewer.`;
  button.setAttribute('aria-pressed', String(appState.showPrecomputedHighlights));
  button.style.setProperty('--precompute-bg', colors.buttonBg || colors.precomputedFill || colors.hoverFill);
  button.style.setProperty('--precompute-hover-bg', colors.buttonHoverBg || colors.precomputedCountyFill || colors.hoverFill);
  button.style.setProperty('--precompute-border', colors.buttonBorder || colors.innerStroke);
  button.style.setProperty('--precompute-text', colors.buttonText || colors.precomputedStroke || colors.hoverStroke);
  button.style.setProperty('--precompute-ring', 'rgba(255, 255, 255, 0.42)');
}

function updateBreadcrumb() {
  const breadcrumb = document.getElementById('breadcrumb');
  breadcrumb.replaceChildren();

  const appendSeparator = () => {
    const separator = document.createElement('span');
    separator.className = 'breadcrumb-separator';
    separator.textContent = '>';
    breadcrumb.appendChild(separator);
  };

  const appendCrumb = (label, onClick, isCurrent = false) => {
    const element = document.createElement(isCurrent ? 'span' : 'button');
    element.textContent = label;
    if (isCurrent) {
      element.className = 'breadcrumb-current';
      element.setAttribute('aria-current', 'page');
    } else {
      element.type = 'button';
      element.className = 'breadcrumb-link';
      element.addEventListener('click', onClick);
    }
    breadcrumb.appendChild(element);
  };

  appendCrumb('Country', () => navigate('country'), appState.zoom === 'country');

  if (appState.state) {
    appendSeparator();
    appendCrumb(
      appState.state,
      () => navigate('state', appState.state, null),
      appState.zoom === 'state'
    );
  }

  if (appState.county) {
    appendSeparator();
    appendCrumb(
      appState.county,
      () => navigate('county', appState.state, appState.county),
      appState.zoom === 'county'
    );
  }

  const stateName = STATE_CODE_TO_NAME[appState.state] || appState.state;
  const retainLabel = `${appState.retainPercentage}% retained`;
  const thresholdLabel = `${analysisLabel()} <= ${appState.errorThreshold.toFixed(2)}, baseline ${topologyBaselineLabel(appState.topologyBaselinePercentage)}`;
  const countryCaption = appState.method === 'all'
    ? 'Country view with state boundaries and all exact points.'
    : appState.method === 'pixel'
      ? 'Country view with state boundaries and viewport-aware pixel occupancy points.'
      : appState.method === 'fps_threshold'
        ? `Country view with state boundaries and FPS threshold sampling (${thresholdLabel}).`
        : `Country view with state boundaries and sampled points (${retainLabel}).`;
  const stateCaption = appState.method === 'all'
    ? `${stateName} boundary with county outlines, hover county names, and all exact points.`
    : appState.method === 'pixel'
      ? `${stateName} boundary with county outlines, hover county names, and viewport-aware pixel occupancy points.`
      : appState.method === 'fps_threshold'
        ? `${stateName} boundary with county outlines, hover county names, and FPS threshold sampling (${thresholdLabel}).`
        : `${stateName} boundary with county outlines, hover county names, and sampled points (${retainLabel}).`;
  const countyCaption = appState.method === 'all'
    ? `${appState.county} boundary with full county points.`
    : appState.method === 'pixel'
      ? `${appState.county} boundary with pixel-sampled county points.`
      : appState.method === 'fps_threshold'
        ? `${appState.county} boundary with FPS threshold sampling (${thresholdLabel}).`
        : `${appState.county} boundary with sampled county points (${retainLabel}).`;

  const caption = appState.zoom === 'country'
    ? countryCaption
    : appState.zoom === 'state'
      ? stateCaption
      : countyCaption;
  document.getElementById('view-caption').textContent = caption + randomPrecomputeNote();
}

async function loadPoints() {
  const requestToken = ++pointsRequestToken;
  setStatus(`Loading ${appState.method === 'fps_threshold' ? 'threshold' : appState.method} points...`);
  const params = new URLSearchParams({
    zoom: appState.zoom,
    method: appState.method,
    state: appState.state || '',
    county: appState.county || '',
    analysis_property: appState.analysisProperty,
    retain_pct: String(appState.retainPercentage),
    error_threshold: appState.errorThreshold.toFixed(2),
    topology_baseline_pct: String(appState.topologyBaselinePercentage),
  });
  appendViewportParams(params);
  try {
    const response = await fetch(`/api/data?${params}`);
    const payload = await response.json();
    if (requestToken !== pointsRequestToken) {
      return false;
    }

    if (payload.static_demo_unavailable) {
      sources.points.clear();
      setStatus(payload.message || 'This static demo does not include that precomputed snapshot.');
      renderStaticDemoMessage(payload);
      return false;
    }

    if (payload.missing_cache) {
      sources.points.clear();
      setStatus(payload.message || 'FPS-threshold data has not been generated for this view.');
      renderFpsThresholdCacheMessage(payload);
      if (payload.generation_status === 'queued' || payload.generation_status === 'running') {
        startFpsGenerationPolling();
      }
      return false;
    }

    const features = (payload.points || []).map((point) => {
      return new ol.Feature({
        geometry: new ol.geom.Point(ol.proj.fromLonLat([Number(point.lng), Number(point.lat)])),
        kind: 'point',
        state: point.State || null,
        county: point.County || null,
        pointId: point.ID || null,
      });
    });

    sources.points.clear();
    sources.points.addFeatures(features);

    const displayedCount = payload.displayed_count ?? payload.count;
    const exactCount = payload.exact_count ?? displayedCount;
    const scopeText = appState.zoom === 'country'
      ? `Country view shows ${displayedCount.toLocaleString()} displayed / ${exactCount.toLocaleString()} exact points.`
      : appState.zoom === 'state'
        ? `State view shows ${displayedCount.toLocaleString()} displayed / ${exactCount.toLocaleString()} exact points for ${appState.state}.`
        : `County view shows ${displayedCount.toLocaleString()} displayed / ${exactCount.toLocaleString()} exact points for ${appState.county}.`;
    setStatus(scopeText);
    return true;
  } catch (error) {
    console.error(error);
    if (requestToken !== pointsRequestToken) {
      return false;
    }
    sources.points.clear();
    setStatus('Point data could not be loaded for the current selection.');
    return false;
  }
}

async function loadCountyLabels() {
  const response = await fetch(`/api/labels/counties?state=${encodeURIComponent(appState.state)}`);
  const payload = await response.json();
  const features = payload.labels.map((row) => {
    return new ol.Feature({
      geometry: new ol.geom.Point(ol.proj.fromLonLat([Number(row.lng), Number(row.lat)])),
      kind: 'county-label',
      county: row.County || row.county,
      label: row.County || row.county,
      count: row.count,
    });
  });
  sources.countyLabels.clear();
  sources.countyLabels.addFeatures(features);
}

async function loadCountyBoundary() {
  const params = new URLSearchParams({
    state: appState.state,
    county: appState.county,
  });
  const response = await fetch(`/api/boundaries/county?${params}`);
  const featureGeoJson = await response.json();
  const feature = new ol.format.GeoJSON().readFeature(featureGeoJson, {
    featureProjection: 'EPSG:3857',
  });
  feature.set('kind', 'county-boundary');
  sources.countyBoundary.clear();
  sources.countyBoundary.addFeature(feature);
}

function prepareSelectedState() {
  const feature = findStateFeature(appState.state);
  sources.selectedState.clear();
  if (feature) {
    sources.selectedState.addFeature(feature.clone());
  }
}

function fitSelectedState() {
  const extent = sources.selectedState.getExtent();
  if (!ol.extent.isEmpty(extent)) {
    map.getView().fit(extent, {
      padding: STATE_FIT_PADDING,
      duration: 250,
      maxZoom: 8,
    });
  }
}

function fitCountyBoundary() {
  const extent = sources.countyBoundary.getExtent();
  if (!ol.extent.isEmpty(extent)) {
    map.getView().fit(extent, {
      padding: COUNTY_FIT_PADDING,
      duration: 250,
      maxZoom: 11,
    });
  }
}

function renderAnalysisLoading() {
  const content = document.getElementById('analysis-content');
  const title = document.getElementById('analysis-title');
  const subtitle = document.getElementById('analysis-subtitle');

  if (appState.analysisProperty === 'topological') {
    title.textContent = 'Structural Comparison';
    subtitle.textContent = 'Computing persistence diagrams and structural distances...';
    content.innerHTML = `
      <div class="analysis-placeholder analysis-loading">
        <div class="analysis-spinner" aria-hidden="true"></div>
        <div class="analysis-loading-text">
          <strong>Calculating structural descriptors...</strong>
          <span>This can take a little longer than statistical or distributional analysis.</span>
        </div>
      </div>
    `;
    return;
  }

  title.textContent = appState.analysisProperty === 'density'
    ? 'Distributional Comparison'
    : `${analysisLabel()} Comparison`;
  subtitle.textContent = 'Updating analysis...';
  content.innerHTML = '<div class="analysis-placeholder">Updating analysis...</div>';
}

async function loadAnalysis() {
  const requestToken = ++analysisRequestToken;
  renderAnalysisLoading();

  const params = new URLSearchParams({
    property: appState.analysisProperty,
    zoom: appState.zoom,
    method: appState.method,
    state: appState.state || '',
    county: appState.county || '',
    retain_pct: String(appState.retainPercentage),
    error_threshold: appState.errorThreshold.toFixed(2),
    topology_baseline_pct: String(appState.topologyBaselinePercentage),
  });
  appendViewportParams(params);

  try {
    const response = await fetch(`/api/analysis?${params}`);
    const payload = await response.json();
    if (requestToken !== analysisRequestToken) {
      return false;
    }
    if (payload.static_demo_unavailable) {
      renderStaticDemoMessage(payload);
      return false;
    }
    if (payload.missing_cache) {
      renderFpsThresholdCacheMessage(payload);
      if (payload.generation_status === 'queued' || payload.generation_status === 'running') {
        startFpsGenerationPolling();
      }
      return false;
    }
    renderAnalysis(payload);
    return true;
  } catch (error) {
    console.error(error);
    if (requestToken !== analysisRequestToken) {
      return false;
    }
    document.getElementById('analysis-title').textContent = 'Analysis Unavailable';
    document.getElementById('analysis-subtitle').textContent = 'This comparison could not be computed.';
    document.getElementById('analysis-content').innerHTML =
      '<div class="analysis-placeholder">Analysis data could not be loaded for the current selection.</div>';
    return false;
  }
}

function getViewportSpecForRequests() {
  if (!map) {
    return null;
  }

  const size = map.getSize();
  if (!size || size.length < 2 || !size[0] || !size[1]) {
    return null;
  }

  const padding = appState.zoom === 'country'
    ? COUNTRY_FIT_PADDING
    : appState.zoom === 'state'
      ? STATE_FIT_PADDING
      : COUNTY_FIT_PADDING;

  return {
    width: size[0],
    height: size[1],
    padding,
  };
}

function appendViewportParams(params) {
  const viewport = getViewportSpecForRequests();
  if (!viewport) {
    return;
  }

  params.set('viewport_width_px', String(viewport.width));
  params.set('viewport_height_px', String(viewport.height));
  params.set('padding_top_px', String(viewport.padding[0]));
  params.set('padding_right_px', String(viewport.padding[1]));
  params.set('padding_bottom_px', String(viewport.padding[2]));
  params.set('padding_left_px', String(viewport.padding[3]));
}

function schedulePixelViewportRefresh() {
  if (!map || appState.method !== 'pixel') {
    return;
  }

  if (pixelViewportRefreshTimer) {
    window.clearTimeout(pixelViewportRefreshTimer);
  }

  pixelViewportRefreshTimer = window.setTimeout(() => {
    pixelViewportRefreshTimer = null;
    loadPoints();
    loadAnalysis();
  }, 220);
}

function currentFpsGenerationRequest() {
  return {
    zoom: appState.zoom,
    state: appState.state || '',
    county: appState.county || '',
  };
}

function stopFpsGenerationPolling() {
  if (fpsGenerationPollTimer) {
    window.clearTimeout(fpsGenerationPollTimer);
    fpsGenerationPollTimer = null;
  }
}

function isCurrentFpsGenerationScope(scope) {
  return scope
    && scope.zoom === appState.zoom
    && (scope.state || '') === (appState.state || '')
    && (scope.county || '') === (appState.county || '');
}

function renderFpsThresholdCacheMessage(payload) {
  const content = document.getElementById('analysis-content');
  const title = document.getElementById('analysis-title');
  const subtitle = document.getElementById('analysis-subtitle');
  const status = payload.status || payload.result || payload || {};
  lastFpsThresholdStatus = status;
  const job = payload.job || status.job || null;
  const generationStatus = payload.generation_status || status.generation_status || job?.status || 'missing';
  const exactCount = Number(status.exact_count ?? payload.exact_count ?? 0);
  const rowCount = Number(status.row_count ?? 0);
  const expectedRows = Math.max(1, Number(status.expected_row_count ?? 100));
  const isRunning = generationStatus === 'queued' || generationStatus === 'running';
  const isDone = generationStatus === 'done' || job?.status === 'done';
  const isError = generationStatus === 'error' || job?.status === 'error';
  const progressPct = isDone ? 100 : Math.max(0, Math.min(100, Math.round((rowCount / expectedRows) * 100)));
  const scopeType = appState.zoom === 'county' ? 'County' : 'State';

  title.textContent = isDone
    ? 'Data precomputation is complete.'
    : `Data is not precomputed for this ${scopeType}.`;
  subtitle.textContent = `${exactCount.toLocaleString()} points`;

  const disabledAttr = isRunning ? ' disabled' : '';
  const generateLabel = isRunning ? 'Generating...' : isError ? 'Try Again' : 'Generate Data';
  const reloadButton = isDone
    ? '<button id="fps-threshold-reload-btn" type="button" class="cache-action-primary">Reload Dashboard</button>'
    : '';
  const generateButton = isDone
    ? ''
    : `<button id="fps-threshold-generate-btn" type="button" class="cache-action-primary"${disabledAttr}>${generateLabel}</button>`;

  content.innerHTML = `
    <div class="analysis-placeholder cache-action-card">
      <div class="cache-action-details">
        <span>${exactCount.toLocaleString()} points</span>
        <span>${progressPct}%</span>
      </div>
      <div class="cache-progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${progressPct}">
        <div class="cache-progress-bar" style="width: ${progressPct}%"></div>
      </div>
      <div class="cache-action-row">
        ${generateButton}
        ${reloadButton}
      </div>
    </div>
  `;

  const generateBtn = document.getElementById('fps-threshold-generate-btn');
  if (generateBtn) {
    generateBtn.addEventListener('click', startFpsThresholdGeneration);
  }
  const reloadBtn = document.getElementById('fps-threshold-reload-btn');
  if (reloadBtn) {
    reloadBtn.addEventListener('click', () => window.location.reload());
  }
}

function renderStaticDemoMessage(payload) {
  const content = document.getElementById('analysis-content');
  const title = document.getElementById('analysis-title');
  const subtitle = document.getElementById('analysis-subtitle');
  const exactCount = Number(payload.exact_count ?? payload.status?.exact_count ?? 0);
  title.textContent = 'Static Demo Snapshot Unavailable';
  subtitle.textContent = exactCount > 0
    ? `${exactCount.toLocaleString()} points in this scope`
    : 'Precomputed snapshots only';
  content.innerHTML = `
    <div class="analysis-placeholder cache-action-card">
      <strong>${payload.message || 'This GitHub Pages demo includes only selected precomputed snapshots.'}</strong>
      <span>Clone the repository and run the local dashboard to generate new state or county FPS-threshold data.</span>
    </div>
  `;
}

async function startFpsThresholdGeneration() {
  const scope = currentFpsGenerationRequest();
  stopFpsGenerationPolling();
  renderFpsThresholdCacheMessage({
    generation_status: 'queued',
    message: 'Data precomputation is queued.',
    status: {
      zoom: scope.zoom,
      state: scope.state,
      county: scope.county,
      scope_label: scope.county ? `${scope.county}, ${scope.state}` : scope.state,
      exact_count: lastFpsThresholdStatus?.exact_count ?? 0,
      row_count: lastFpsThresholdStatus?.row_count ?? 0,
      expected_row_count: lastFpsThresholdStatus?.expected_row_count ?? 100,
    },
  });
  setStatus('Data precomputation is queued.');

  try {
    const response = await fetch('/api/fps-threshold/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(scope),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.message || 'FPS-threshold generation could not be started.');
    }
    renderFpsThresholdCacheMessage(payload);
    setStatus(payload.message || 'Data precomputation is queued.');
    if (payload.generation_status === 'queued' || payload.generation_status === 'running') {
      startFpsGenerationPolling(scope);
    }
  } catch (error) {
    console.error(error);
    renderFpsThresholdCacheMessage({
      generation_status: 'error',
      message: error.message || 'Data precomputation could not be started.',
      status: {
        zoom: scope.zoom,
        state: scope.state,
        county: scope.county,
        scope_label: scope.county ? `${scope.county}, ${scope.state}` : scope.state,
      },
    });
    setStatus('Data precomputation could not be started.');
  }
}

function startFpsGenerationPolling(scope = currentFpsGenerationRequest()) {
  stopFpsGenerationPolling();
  const poll = async () => {
    if (!isCurrentFpsGenerationScope(scope)) {
      stopFpsGenerationPolling();
      return;
    }
    const params = new URLSearchParams(scope);
    try {
      const response = await fetch(`/api/fps-threshold/status?${params}`);
      const payload = await response.json();
      if (!isCurrentFpsGenerationScope(scope)) {
        return;
      }
      renderFpsThresholdCacheMessage(payload);
      const generationStatus = payload.generation_status || payload.job?.status;
      if (generationStatus === 'queued' || generationStatus === 'running') {
        fpsGenerationPollTimer = window.setTimeout(poll, 5000);
        setStatus(payload.job?.message || 'Data precomputation is running.');
      } else if (generationStatus === 'done') {
        stopFpsGenerationPolling();
        setStatus('Data precomputation is complete.');
      } else if (generationStatus === 'error') {
        stopFpsGenerationPolling();
        setStatus(payload.job?.message || 'Data precomputation failed.');
      }
    } catch (error) {
      console.error(error);
      if (isCurrentFpsGenerationScope(scope)) {
        fpsGenerationPollTimer = window.setTimeout(poll, 8000);
      }
    }
  };

  fpsGenerationPollTimer = window.setTimeout(poll, 1200);
}

function renderAnalysis(payload) {
  const content = document.getElementById('analysis-content');
  const title = document.getElementById('analysis-title');
  const subtitle = document.getElementById('analysis-subtitle');

  if (payload?.missing_cache) {
    renderFpsThresholdCacheMessage(payload);
    return;
  }

  if (!payload || payload.available === false) {
    title.textContent = 'Analysis Unavailable';
    subtitle.textContent = 'This comparison could not be computed.';
    content.innerHTML = '<div class="analysis-placeholder">Analysis data is unavailable for the current selection.</div>';
    return;
  }

  const displayedCount = payload.displayed_count ?? payload.count ?? null;
  const exactCount = payload.exact_count ?? displayedCount;
  const countSuffix = displayedCount !== null && exactCount !== null
    ? ` Displayed ${Number(displayedCount).toLocaleString()} / ${Number(exactCount).toLocaleString()} exact points.`
    : '';

  if (appState.analysisProperty === 'statistical') {
    title.textContent = 'Statistical Comparison';
    subtitle.textContent = 'Absolute differences between original and reduced summary metrics.' + countSuffix;
    renderStatisticalAnalysis(content, payload);
    return;
  }

  if (appState.analysisProperty === 'density') {
    title.textContent = 'Distributional Comparison';
    subtitle.textContent = 'Residual density grid: warm cells are underrepresented; cool cells are overrepresented.' + countSuffix;
    renderDensityAnalysis(content, payload);
    return;
  }

  title.textContent = 'Structural Comparison';
  const baselinePct = Number(payload.baseline_percentage);
  const currentPct = Number(payload.current_percentage);
  const payloadBaselineLabel = payload.baseline_label || (Number.isFinite(baselinePct) ? `${baselinePct}% FPS` : null);
  const payloadCurrentLabel = payload.current_label || (Number.isFinite(currentPct) ? `${currentPct}% FPS` : null);
  const selectedDistanceBaselineLabel = DISTANCE_BASELINE_LABELS[appState.topologyBaselinePercentage] || null;
  const staleDistanceBaselineWarning = selectedDistanceBaselineLabel
    && payload.baseline_key !== appState.topologyBaselinePercentage
    && payload.baseline_label !== selectedDistanceBaselineLabel
    ? ' Backend returned a percentage baseline; restart the dashboard server to apply the distance-baseline backend.'
    : '';
  const completenessSuffix = payload.complete === false
    ? ' Country precompute is still in progress; showing the saved checkpoints available so far.'
    : '';
  const topologyScopeSuffix = payloadBaselineLabel && payloadCurrentLabel
    ? ` Baseline ${payloadBaselineLabel} vs current ${payloadCurrentLabel}.`
    : ' Persistence diagrams for original and reduced points.';
  subtitle.textContent = topologyScopeSuffix + countSuffix + completenessSuffix + staleDistanceBaselineWarning;
  renderTopologicalAnalysis(content, payload);
}

function renderStatisticalAnalysis(container, payload) {
  container.innerHTML = `
    <div class="analysis-grid">
      <div id="stat-figure" class="analysis-figure"></div>
      <div class="analysis-note" id="stat-summary"></div>
    </div>
  `;

  const labels = ['Mean Lat', 'Mean Lng', 'Std Lat', 'Std Lng', 'Correlation'];
  const keys = ['mean_lat', 'mean_lng', 'std_lat', 'std_lng', 'correlation'];
  const rows = keys.map((key, index) => {
    const difference = Math.abs(Number(payload.error?.[key] ?? 0));
    return {
      key,
      label: labels[index],
      difference: Number.isFinite(difference) ? difference : 0,
    };
  }).sort((a, b) => a.difference - b.difference);

  const maxDifference = Math.max(1e-9, ...rows.map((row) => row.difference));
  const xMax = maxDifference * 1.18;
  const barColor = 'rgba(46, 111, 176, 0.82)';

  Plotly.newPlot(
    'stat-figure',
    [
      {
        type: 'bar',
        orientation: 'h',
        name: 'Difference',
        x: rows.map((row) => row.difference),
        y: rows.map((row) => row.label),
        marker: {
          color: barColor,
          line: { color: 'rgba(26, 77, 128, 0.34)', width: 1 },
        },
        hovertemplate: '%{y}<br>Absolute difference: %{x:.4g}<extra></extra>',
      },
    ],
    {
      margin: { l: 92, r: 26, t: 18, b: 44 },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      showlegend: false,
      xaxis: {
        title: 'Absolute difference',
        range: [0, xMax],
        zeroline: true,
        zerolinecolor: 'rgba(16, 42, 67, 0.28)',
        gridcolor: 'rgba(16, 42, 67, 0.08)',
      },
      yaxis: {
        automargin: true,
      },
      bargap: 0.34,
      annotations: [
        {
          x: 0,
          y: 1.12,
          xref: 'paper',
          yref: 'paper',
          text: 'Longer bar = larger absolute difference',
          showarrow: false,
          align: 'left',
          font: { size: 12, color: '#486581' },
        },
      ],
    },
    { responsive: true, displaylogo: false }
  );

  const highest = rows[rows.length - 1];
  const lowest = rows[0];
  document.getElementById('stat-summary').innerHTML = `
    <div class="stat-difference-summary">
      <span><strong>Largest absolute difference:</strong> ${highest?.label || 'N/A'} (${formatNumber(highest?.difference)})</span>
      <span><strong>Smallest absolute difference:</strong> ${lowest?.label || 'N/A'} (${formatNumber(lowest?.difference)})</span>
    </div>
  `;
}
function renderDensityAnalysis(container, payload) {
  container.innerHTML = `
    <div class="analysis-grid">
      <div id="density-figure" class="analysis-figure tall"></div>
      <div class="analysis-metrics density-summary" id="density-summary"></div>
    </div>
  `;

  const differenceDensity = numericGrid(payload.difference_density);
  const latCenters = binCenters(payload.lat_edges);
  const lngCenters = binCenters(payload.lng_edges);
  const flatDifference = differenceDensity.flat().filter((value) => Number.isFinite(value));
  const maxAbsDifference = Math.max(1e-12, ...flatDifference.map((value) => Math.abs(value)));
  const l1Difference = flatDifference.reduce((sum, value) => sum + Math.abs(value), 0);
  const densityOverlap = Math.max(0, Math.min(1, 1 - (l1Difference / 2)));
  const positiveMass = flatDifference
    .filter((value) => value > 0)
    .reduce((sum, value) => sum + value, 0);
  const negativeMass = Math.abs(flatDifference
    .filter((value) => value < 0)
    .reduce((sum, value) => sum + value, 0));
  const materialThreshold = maxAbsDifference * 0.10;
  const highChangeCells = flatDifference.filter((value) => Math.abs(value) >= materialThreshold).length;
  const totalCells = Math.max(1, flatDifference.length);

  Plotly.newPlot(
    'density-figure',
    [
      {
        z: differenceDensity,
        x: lngCenters,
        y: latCenters,
        type: 'heatmap',
        zmin: -maxAbsDifference,
        zmax: maxAbsDifference,
        zmid: 0,
        colorscale: [
          [0, '#2f6f9f'],
          [0.46, '#e8f1f6'],
          [0.5, '#fbfbf8'],
          [0.54, '#f6e8df'],
          [1, '#b75d3a'],
        ],
        colorbar: {
          title: { text: 'Original - displayed', side: 'right' },
          thickness: 11,
          len: 0.82,
          outlinewidth: 0,
        },
        hovertemplate: [
          'Longitude: %{x:.3f}',
          'Latitude: %{y:.3f}',
          'Residual: %{z:.4g}',
          '<extra></extra>',
        ].join('<br>'),
      },
    ],
    {
      margin: { l: 58, r: 82, t: 58, b: 50 },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      xaxis: {
        title: 'Longitude',
        zeroline: false,
        gridcolor: 'rgba(16, 42, 67, 0.08)',
        tickfont: { size: 10 },
      },
      yaxis: {
        title: 'Latitude',
        zeroline: false,
        gridcolor: 'rgba(16, 42, 67, 0.08)',
        tickfont: { size: 10 },
        scaleanchor: 'x',
        scaleratio: 1,
      },
      annotations: [
        {
          text: '<b>Density Residual</b>',
          x: 0,
          y: 1.14,
          xref: 'paper',
          yref: 'paper',
          xanchor: 'left',
          showarrow: false,
          font: { size: 13, color: '#102a43' },
        },
        {
          text: 'Warm = original has more density; cool = displayed has more density',
          x: 0,
          y: 1.06,
          xref: 'paper',
          yref: 'paper',
          xanchor: 'left',
          showarrow: false,
          font: { size: 11, color: '#486581' },
        },
      ],
    },
    { responsive: true, displaylogo: false }
  );

  const overlapPct = densityOverlap * 100;
  const highChangePct = (highChangeCells / totalCells) * 100;
  document.getElementById('density-summary').innerHTML = `
    <div class="density-summary-grid">
      <div class="density-summary-card">
        <span>Density Overlap</span>
        <strong>${formatNumber(overlapPct)}%</strong>
      </div>
      <div class="density-summary-card">
        <span>Largest Local Gap</span>
        <strong>${formatNumber(maxAbsDifference)}</strong>
      </div>
      <div class="density-summary-card">
        <span>Underrepresented Mass</span>
        <strong>${formatNumber(positiveMass)}</strong>
      </div>
      <div class="density-summary-card">
        <span>Overrepresented Mass</span>
        <strong>${formatNumber(negativeMass)}</strong>
      </div>
    </div>
    <div class="density-summary-caption">
      <span><strong>L2 norm:</strong> ${formatNumber(payload.metrics?.l2_norm)}</span>
      <span><strong>Linf distance:</strong> ${formatNumber(payload.metrics?.linf_distance)}</span>
      <span><strong>High-change cells:</strong> ${highChangeCells} / ${totalCells} (${formatNumber(highChangePct)}%)</span>
    </div>
  `;
}

function numericGrid(grid) {
  return (grid || []).map((row) => (row || []).map((value) => {
    const numberValue = Number(value);
    return Number.isFinite(numberValue) ? numberValue : 0;
  }));
}

function binCenters(edges) {
  const values = (edges || []).map((value) => Number(value)).filter((value) => Number.isFinite(value));
  if (values.length < 2) {
    return [];
  }
  return values.slice(0, -1).map((value, index) => (value + values[index + 1]) / 2);
}

function renderTopologicalAnalysis(container, payload) {
  container.innerHTML = `
    <div class="analysis-grid">
      <div id="topology-figure" class="analysis-figure tall"></div>
      <div class="analysis-metrics" id="topology-metrics"></div>
    </div>
  `;

  const traces = [];
  const originalDiagram = combinedDiagram(payload.original.h0, payload.original.h1);
  const reducedDiagram = combinedDiagram(payload.reduced.h0, payload.reduced.h1);
  const diagramRange = persistenceDiagramRange([originalDiagram, reducedDiagram]);
  const reducedTitle = appState.method === 'fps_threshold'
    ? 'FPS PD'
    : 'Reduced PD';

  traces.push(diagonalTrace(diagramRange, 'x', 'y'));
  traces.push(diagramTrace(originalDiagram, 'All-Points Persistence Diagram', 'x', 'y'));
  traces.push(diagonalTrace(diagramRange, 'x2', 'y2'));
  traces.push(diagramTrace(reducedDiagram, reducedTitle, 'x2', 'y2'));

  Plotly.newPlot(
    'topology-figure',
    traces,
    {
      margin: { l: 54, r: 24, t: 44, b: 48 },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      grid: { rows: 1, columns: 2, pattern: 'independent' },
      xaxis: { ...persistenceAxis('Birth', diagramRange), domain: [0.0, 0.43] },
      yaxis: persistenceAxis('Death', diagramRange, 'x'),
      xaxis2: { ...persistenceAxis('Birth', diagramRange), domain: [0.57, 1.0] },
      yaxis2: persistenceAxis('Death', diagramRange, 'x2'),
      annotations: [
        { text: 'All-Points PD', x: 0.215, y: 1.12, xref: 'paper', yref: 'paper', showarrow: false, font: { size: 12 } },
        { text: reducedTitle, x: 0.785, y: 1.12, xref: 'paper', yref: 'paper', showarrow: false, font: { size: 12 } },
      ],
      showlegend: false,
    },
    { responsive: true, displaylogo: false }
  );

  const baselineDistances = payload.baseline_distances || null;
  const currentDistances = payload.current_distances || payload.distances || {};
  const normalizedDistances = payload.normalized_distances || null;
  const baselinePct = Number(payload.baseline_percentage);
  const currentPct = Number(payload.current_percentage);
  const payloadBaselineLabel = payload.baseline_label || null;
  const payloadCurrentLabel = payload.current_label || null;
  const metricDefs = [
    ['Bottleneck H0', 'bottleneck_h0'],
    ['Bottleneck H1', 'bottleneck_h1'],
    ['Wasserstein H0', 'wasserstein_h0'],
    ['Wasserstein H1', 'wasserstein_h1'],
  ];
  const metricsHtml = baselineDistances && normalizedDistances
    ? (() => {
        const metricRows = metricDefs.map(([label, key]) => `
          <tr>
            <th>${label}</th>
            <td>${formatNumber(baselineDistances[key])}</td>
            <td>${formatNumber(currentDistances[key])}</td>
            <td>${formatNumber(normalizedDistances[key])}</td>
          </tr>
        `).join('');
        const baselineHeader = payloadBaselineLabel
          ? `Baseline ${payloadBaselineLabel}`
          : Number.isFinite(baselinePct)
            ? `Baseline ${baselinePct}%`
            : 'Baseline';
        const currentHeader = payloadCurrentLabel
          ? `Current ${payloadCurrentLabel}`
          : Number.isFinite(currentPct)
            ? `Current ${currentPct}%`
            : 'Current';
        return `
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>${baselineHeader}</th>
                <th>${currentHeader}</th>
                <th>Normalized</th>
              </tr>
            </thead>
            <tbody>
              ${metricRows}
            </tbody>
          </table>
        `;
      })()
    : `
      <table>
        <tbody>
          ${metricDefs.map(([label, key]) => `
            <tr><th>${label}</th><td>${formatNumber(currentDistances[key])}</td></tr>
          `).join('')}
        </tbody>
      </table>
    `;
  document.getElementById('topology-metrics').innerHTML = metricsHtml;
}

function diagramTrace(diagram, name, xaxis, yaxis) {
  return {
    type: 'scattergl',
    mode: 'markers',
    name,
    x: diagram.x,
    y: diagram.y,
    customdata: diagram.family,
    xaxis,
    yaxis,
    marker: {
      color: PERSISTENCE_DIAGRAM_COLOR,
      size: 5,
      opacity: 0.82,
    },
    hovertemplate: '%{customdata}<br>Birth %{x:.4g}<br>Death %{y:.4g}<extra></extra>',
  };
}

function diagonalTrace(range, xaxis, yaxis) {
  return {
    type: 'scatter',
    mode: 'lines',
    x: range,
    y: range,
    xaxis,
    yaxis,
    line: {
      color: PERSISTENCE_DIAGONAL_COLOR,
      width: 1.5,
      dash: 'dash',
    },
    hoverinfo: 'skip',
  };
}

function combinedDiagram(h0Points, h1Points) {
  const h0 = (h0Points || []).map((row) => [row[0], row[1], 'H0']);
  const h1 = (h1Points || []).map((row) => [row[0], row[1], 'H1']);
  const rows = h0.concat(h1).filter((row) => Number.isFinite(Number(row[0])) && Number.isFinite(Number(row[1])));
  return {
    x: rows.map((row) => Number(row[0])),
    y: rows.map((row) => Number(row[1])),
    family: rows.map((row) => row[2]),
  };
}

function persistenceDiagramRange(diagrams) {
  let minValue = 0;
  let maxValue = 1e-6;
  let foundValue = false;
  diagrams.forEach((diagram) => {
    [diagram.x, diagram.y].forEach((values) => {
      values.forEach((value) => {
        if (!Number.isFinite(value)) {
          return;
        }
        foundValue = true;
        minValue = Math.min(minValue, value);
        maxValue = Math.max(maxValue, value);
      });
    });
  });
  if (!foundValue) {
    return [-0.05, 1.05];
  }
  const span = Math.max(maxValue - minValue, maxValue, 1e-6);
  const padding = span * 0.06;
  return [minValue - padding, maxValue + padding];
}

function persistenceAxis(title, range, scaleAnchor = null) {
  const axis = {
    title,
    range,
    zeroline: false,
    showgrid: true,
    gridcolor: 'rgba(16, 42, 67, 0.09)',
    linecolor: 'rgba(16, 42, 67, 0.72)',
    mirror: true,
    ticks: 'outside',
  };
  if (scaleAnchor) {
    axis.scaleanchor = scaleAnchor;
    axis.scaleratio = 1;
  }
  return axis;
}

function handlePointerMove(event) {
  if (!map || appState.zoom !== 'state' || event.dragging) {
    clearCountyHover();
    return;
  }

  const hovered = map.forEachFeatureAtPixel(
    event.pixel,
    (feature) => {
      const kind = feature.get('kind');
      if (kind === 'point' && feature.get('county')) {
        return { county: feature.get('county') };
      }
      if (kind === 'state-county-boundary' && feature.get('county')) {
        return { county: feature.get('county') };
      }
      return null;
    },
    { hitTolerance: 6 }
  );

  if (!hovered || !hovered.county) {
    clearCountyHover();
    return;
  }

  map.getTargetElement().style.cursor = 'pointer';
  showCountyHover(hovered.county, event.pixel);
}

function showCountyHover(countyName, pixel) {
  const countyFeature = sources.stateCountyBoundaries
    .getFeatures()
    .find((feature) => feature.get('county') === countyName);

  sources.hoveredCountyBoundary.clear();
  if (countyFeature) {
    sources.hoveredCountyBoundary.addFeature(countyFeature.clone());
  }

  if (!countyHoverTooltip) {
    return;
  }

  countyHoverTooltip.textContent = countyName;
  countyHoverTooltip.style.left = `${pixel[0] + 14}px`;
  countyHoverTooltip.style.top = `${pixel[1] + 14}px`;
  countyHoverTooltip.hidden = false;
}

function clearCountyHover() {
  sources.hoveredCountyBoundary.clear();
  if (map) {
    map.getTargetElement().style.cursor = '';
  }
  if (countyHoverTooltip) {
    countyHoverTooltip.hidden = true;
  }
}

function handleMapClick(event) {
  map.forEachFeatureAtPixel(
    event.pixel,
    (feature) => {
      const kind = feature.get('kind');
      if (appState.zoom === 'country') {
        if (kind === 'point' && feature.get('state')) {
          navigate('state', feature.get('state'), null);
          return true;
        }
        if ((kind === 'state-label' || kind === 'state-boundary') && feature.get('stateCode')) {
          navigate('state', feature.get('stateCode'), null);
          return true;
        }
      } else if (appState.zoom === 'state') {
        if (kind === 'state-county-boundary' && feature.get('county')) {
          navigate('county', appState.state, feature.get('county'));
          return true;
        }
        if (kind === 'point' && feature.get('county')) {
          navigate('county', appState.state, feature.get('county'));
          return true;
        }
      }
      return false;
    },
    {
      hitTolerance: 6,
    }
  );
}

function findStateFeature(stateCode) {
  const features = sources.states.getFeatures();
  return features.find((feature) => feature.get('stateCode') === stateCode) || null;
}

function makeStateLabelFeature(feature) {
  const center = stateLabelCoordinate(feature);
  return new ol.Feature({
    geometry: new ol.geom.Point(center),
    kind: 'state-label',
    stateCode: feature.get('stateCode'),
    label: feature.get('stateCode'),
  });
}

function stateLabelCoordinate(feature) {
  const geometry = feature.getGeometry();
  if (!geometry) {
    return [0, 0];
  }

  const geometryType = geometry.getType();
  if (geometryType === 'Polygon' && typeof geometry.getInteriorPoint === 'function') {
    return geometry.getInteriorPoint().getCoordinates().slice(0, 2);
  }
  if (geometryType === 'MultiPolygon' && typeof geometry.getInteriorPoints === 'function') {
    const coordinates = geometry.getInteriorPoints().getCoordinates();
    if (coordinates.length) {
      const best = coordinates.reduce((winner, current) => {
        const currentWidth = Number(current[2] || 0);
        const winnerWidth = Number(winner[2] || 0);
        return currentWidth > winnerWidth ? current : winner;
      }, coordinates[0]);
      return best.slice(0, 2);
    }
  }

  return ol.extent.getCenter(geometry.getExtent());
}

function codeFromFeature(feature) {
  const rawId = feature.getId() ?? feature.get('id');
  const fips = String(rawId).padStart(2, '0');
  return STATE_FIPS_TO_CODE[fips] || null;
}

function countyFipsFromFeature(feature) {
  const rawId = feature.getId() ?? feature.get('id');
  const digits = String(rawId).replace(/\D/g, '');
  return digits ? digits.padStart(5, '0') : null;
}

function createTextStyle(text, options) {
  return new ol.style.Style({
    text: new ol.style.Text({
      text,
      font: options.font,
      fill: new ol.style.Fill({ color: options.fill }),
      stroke: new ol.style.Stroke({
        color: options.stroke,
        width: options.strokeWidth,
      }),
      overflow: true,
    }),
  });
}

function createLabelBadgeStyle(text) {
  return new ol.style.Style({
    image: new ol.style.Circle({
      radius: 0.1,
      fill: new ol.style.Fill({ color: 'rgba(0,0,0,0)' }),
    }),
    text: new ol.style.Text({
      text,
      font: '600 11px "Segoe UI", sans-serif',
      fill: new ol.style.Fill({ color: '#102a43' }),
      backgroundFill: new ol.style.Fill({ color: 'rgba(255,255,255,0.94)' }),
      backgroundStroke: new ol.style.Stroke({ color: '#cfd8dc', width: 1 }),
      padding: [3, 5, 3, 5],
      overflow: true,
      offsetY: -10,
    }),
  });
}

function setStatus(text) {
  document.getElementById('map-status').textContent = text;
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'N/A';
  }
  if (!Number.isFinite(Number(value))) {
    return 'Inf';
  }
  return Number(value).toFixed(4);
}

window.addEventListener('load', initApp);













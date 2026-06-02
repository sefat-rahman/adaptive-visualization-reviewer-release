(function () {
  'use strict';

  const originalFetch = window.fetch.bind(window);
  const manifestPromise = originalFetch('data/manifest.json').then((response) => response.json());
  const VIEWPORT_KEYS = new Set([
    'viewport_width_px',
    'viewport_height_px',
    'padding_top_px',
    'padding_right_px',
    'padding_bottom_px',
    'padding_left_px',
  ]);
  const FPS_BASELINES = new Set([
    'country_1pct_distance',
    'county_10pct_distance',
    'county_20pct_distance',
  ]);

  function jsonResponse(payload, status = 200) {
    return new Response(JSON.stringify(payload), {
      status,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  function buildKey(pathname, pairs) {
    const params = new URLSearchParams();
    pairs.forEach(([key, value]) => params.set(key, value == null ? '' : String(value)));
    const query = params.toString();
    return query ? `${pathname}?${query}` : pathname;
  }

  function normalizeThreshold(value) {
    const numberValue = Number(value);
    return Number.isFinite(numberValue) ? numberValue.toFixed(2) : '0.50';
  }

  function normalizeBaseline(value) {
    return FPS_BASELINES.has(value) ? value : 'county_20pct_distance';
  }

  function normalizeApiKey(url) {
    const pathname = url.pathname.replace(/\/+$/, '') || '/';
    const params = new URLSearchParams(url.search);
    VIEWPORT_KEYS.forEach((key) => params.delete(key));

    if (pathname === '/api/meta') {
      return pathname;
    }
    if (pathname === '/api/boundaries/us/nation'
      || pathname === '/api/boundaries/us/states'
      || pathname === '/api/boundaries/us/counties') {
      return pathname;
    }
    if (pathname === '/api/labels/counties') {
      return buildKey(pathname, [['state', params.get('state') || '']]);
    }
    if (pathname === '/api/boundaries/county') {
      return buildKey(pathname, [
        ['state', params.get('state') || ''],
        ['county', params.get('county') || ''],
      ]);
    }
    if (pathname === '/api/fps-threshold/status') {
      return buildKey(pathname, [
        ['zoom', params.get('zoom') || 'state'],
        ['state', params.get('state') || ''],
        ['county', params.get('county') || ''],
      ]);
    }
    if (pathname === '/api/data') {
      const method = params.get('method') || 'pixel';
      const pairs = [
        ['zoom', params.get('zoom') || 'country'],
        ['method', method],
        ['state', params.get('state') || ''],
        ['county', params.get('county') || ''],
      ];
      if (method === 'random') {
        pairs.push(['retain_pct', params.get('retain_pct') || '50']);
      }
    if (method === 'fps_threshold') {
      pairs.push(['error_threshold', normalizeThreshold(params.get('error_threshold'))]);
      pairs.push(['topology_baseline_pct', normalizeBaseline(params.get('topology_baseline_pct'))]);
    }
      return buildKey(pathname, pairs);
    }
    if (pathname === '/api/analysis') {
      const method = params.get('method') || 'pixel';
      const pairs = [
        ['property', params.get('property') || 'statistical'],
        ['zoom', params.get('zoom') || 'country'],
        ['method', method],
        ['state', params.get('state') || ''],
        ['county', params.get('county') || ''],
      ];
      if (method === 'random') {
        pairs.push(['retain_pct', params.get('retain_pct') || '50']);
      }
    if (method === 'fps_threshold') {
      pairs.push(['error_threshold', normalizeThreshold(params.get('error_threshold'))]);
      pairs.push(['topology_baseline_pct', normalizeBaseline(params.get('topology_baseline_pct'))]);
    }
      return buildKey(pathname, pairs);
    }
    return pathname;
  }

  function staticUnavailablePayload(url) {
    const params = new URLSearchParams(url.search);
    return {
      available: false,
      static_demo_unavailable: true,
      generation_supported: false,
      generation_status: 'static_demo',
      exact_count: 0,
      message: 'This GitHub Pages demo includes only selected precomputed snapshots. Clone the repository and run the local dashboard to generate additional views.',
      status: {
        zoom: params.get('zoom') || '',
        state: params.get('state') || '',
        county: params.get('county') || '',
        exact_count: 0,
        row_count: 0,
        expected_row_count: 100,
      },
    };
  }

  window.fetch = async function staticApiFetch(resource, options) {
    const requestUrl = typeof resource === 'string' ? resource : resource.url;
    const url = new URL(requestUrl, window.location.href);
    if (!url.pathname.startsWith('/api/')) {
      return originalFetch(resource, options);
    }

    if (url.pathname === '/api/fps-threshold/generate') {
      return jsonResponse(staticUnavailablePayload(url), 409);
    }

    const manifest = await manifestPromise;
    const key = normalizeApiKey(url);
    const snapshotPath = manifest.routes[key];
    if (!snapshotPath) {
      return jsonResponse(staticUnavailablePayload(url), 404);
    }

    return originalFetch(`data/${snapshotPath}`, { cache: 'force-cache' });
  };
}());

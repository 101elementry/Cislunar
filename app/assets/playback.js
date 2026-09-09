/* Browser-side playback of the 3D scene.

   The server draws the scene once at one instant.  This script runs the
   clock: every animation frame it moves each spacecraft's marker, halo
   and comet trail along the path that is already drawn, and moves the
   Earth and Moon along their paths in the inertial views, so playback
   looks like video instead of a stop-motion redraw.  Speed is mission
   time per real second.  When playback stops the reached sample is
   written to the timeline slider and the server redraws that instant.

   Traces are found by the `meta.role` the figure builder attaches:
   path, trail, halo, marker (per spacecraft) and body, bodypath (per
   body).  The layout's `meta` holds the clock: epoch, time step,
   number of samples and the index the figure was drawn at.

   Moving a trace through Plotly.restyle re-runs Plotly's whole layout
   pipeline (about 200 ms for this scene), far too slow for video.  The
   WebGL trace objects Plotly keeps under the scene are updated directly
   instead, which costs a few milliseconds; restyle remains the fallback
   if a Plotly version hides those objects.  Bodies that move in the
   chosen frame are refreshed every fourth frame because their meshes
   are the expensive part; a body at rest is left alone. */

(function () {
  var state = {playing: false, hoursPerSecond: 6.0, index: 0, lastFrame: null, handle: null,
               caches: {}, frameCount: 0};

  function graphs() {
    var found = [];
    ["view-3d", "view-3d-b"].forEach(function (id) {
      var holder = document.getElementById(id);
      var graph = holder && holder.querySelector(".js-plotly-plot");
      if (graph && graph._fullData && graph._fullLayout && graph._fullLayout.meta
          && graph._fullLayout.meta.n_samples && graph.offsetParent !== null) {
        found.push(graph);
      }
    });
    return found;
  }

  function toArray(values) { return Array.prototype.slice.call(values); }

  // Per-graph lookup of the traces to move and their source arrays.
  // Keyed on graph.data, which Plotly.react replaces when the server
  // sends a new figure but restyle and the direct updates leave alone,
  // so the cached originals stay the originals.
  function cacheFor(graph) {
    var key = graph.parentElement.id;
    var cache = state.caches[key];
    if (cache && cache.data === graph.data) { return cache; }
    cache = {data: graph.data, spacecraft: {}, bodies: {}};
    graph._fullData.forEach(function (trace) {
      var meta = trace.meta || {};
      if (meta.role === "path") {
        cache.spacecraft[meta.spacecraft] = cache.spacecraft[meta.spacecraft] || {};
        cache.spacecraft[meta.spacecraft].path = {stride: meta.stride || 1,
                                                  x: toArray(trace.x), y: toArray(trace.y), z: toArray(trace.z)};
      } else if (meta.role === "trail" || meta.role === "halo" || meta.role === "marker") {
        cache.spacecraft[meta.spacecraft] = cache.spacecraft[meta.spacecraft] || {};
        cache.spacecraft[meta.spacecraft][meta.role] = {uid: trace.uid, samples: meta.samples || 0};
      } else if (meta.role === "bodypath") {
        cache.bodies[meta.body] = cache.bodies[meta.body] || {};
        cache.bodies[meta.body].path = {stride: meta.stride || 1,
                                        x: toArray(trace.x), y: toArray(trace.y), z: toArray(trace.z)};
      } else if (meta.role === "body") {
        cache.bodies[meta.body] = cache.bodies[meta.body] || {};
        cache.bodies[meta.body].surface = {uid: trace.uid,
                                           x: trace.x.map(toArray), y: trace.y.map(toArray), z: trace.z.map(toArray)};
      }
    });
    // The body's drawn position corresponds to the figure's own index;
    // moving it later is an offset from there.
    var drawnIndex = graph._fullLayout.meta.index || 0;
    Object.keys(cache.bodies).forEach(function (name) {
      var body = cache.bodies[name];
      if (body.path && body.surface) {
        body.origin = interpolate(body.path, drawnIndex);
        // A body at rest in this frame (the centre body of a body-centred
        // view) has a path of one repeated point; skip its mesh updates.
        var span = Math.max(Math.max.apply(null, body.path.x) - Math.min.apply(null, body.path.x),
                            Math.max.apply(null, body.path.y) - Math.min.apply(null, body.path.y));
        body.moving = span > 1e-9;
      }
    });
    state.caches[key] = cache;
    return cache;
  }

  // Position along a subsampled path at a fractional full-resolution index.
  function interpolate(path, sampleIndex) {
    var position = sampleIndex / path.stride;
    var n = path.x.length;
    var low = Math.max(0, Math.min(n - 1, Math.floor(position)));
    var high = Math.min(n - 1, low + 1);
    var f = Math.max(0, Math.min(1, position - low));
    return [path.x[low] + (path.x[high] - path.x[low]) * f,
            path.y[low] + (path.y[high] - path.y[low]) * f,
            path.z[low] + (path.z[high] - path.z[low]) * f];
  }

  function fullTrace(graph, uid) {
    for (var i = 0; i < graph._fullData.length; i++) {
      if (graph._fullData[i].uid === uid) { return {index: i, trace: graph._fullData[i]}; }
    }
    return null;
  }

  // Update one trace's coordinates through the WebGL object if Plotly
  // exposes it; otherwise queue a restyle.
  function moveTrace(graph, uid, coords, pending) {
    var found = fullTrace(graph, uid);
    if (!found) { return; }
    var scene = graph._fullLayout.scene && graph._fullLayout.scene._scene;
    var object = scene && scene.traces && scene.traces[uid];
    if (object && object.update) {
      object.update(Object.assign({}, found.trace, coords));
      pending.redraw = scene;
    } else {
      pending.indices.push(found.index);
      pending.x.push(coords.x); pending.y.push(coords.y); pending.z.push(coords.z);
    }
  }

  function draw(graph, sampleIndex, includeBodies) {
    var cache = cacheFor(graph);
    var pending = {indices: [], x: [], y: [], z: [], redraw: null};
    Object.keys(cache.spacecraft).forEach(function (name) {
      var entry = cache.spacecraft[name];
      if (!entry.path) { return; }
      var point = interpolate(entry.path, sampleIndex);
      ["halo", "marker"].forEach(function (role) {
        if (entry[role]) { moveTrace(graph, entry[role].uid, {x: [point[0]], y: [point[1]], z: [point[2]]}, pending); }
      });
      if (entry.trail && entry.trail.samples > 0) {
        var count = entry.trail.samples + 1;
        var tx = new Array(count), ty = new Array(count), tz = new Array(count);
        for (var k = 0; k < count; k++) {
          var p = interpolate(entry.path, Math.max(0, sampleIndex - entry.trail.samples + k));
          tx[k] = p[0]; ty[k] = p[1]; tz[k] = p[2];
        }
        moveTrace(graph, entry.trail.uid, {x: tx, y: ty, z: tz}, pending);
      }
    });
    if (includeBodies) {
      Object.keys(cache.bodies).forEach(function (name) {
        var body = cache.bodies[name];
        if (!body.path || !body.surface || !body.origin || !body.moving) { return; }
        var here = interpolate(body.path, sampleIndex);
        var dx = here[0] - body.origin[0], dy = here[1] - body.origin[1], dz = here[2] - body.origin[2];
        var shift = function (rows, delta) {
          return rows.map(function (row) { return row.map(function (v) { return v + delta; }); });
        };
        moveTrace(graph, body.surface.uid,
                  {x: shift(body.surface.x, dx), y: shift(body.surface.y, dy), z: shift(body.surface.z, dz)}, pending);
      });
    }
    if (pending.redraw) { pending.redraw.glplot.redraw(); }
    if (pending.indices.length > 0) { Plotly.restyle(graph, {x: pending.x, y: pending.y, z: pending.z}, pending.indices); }
  }

  function pad(value) { return (value < 10 ? "0" : "") + value; }

  function readout(graph, sampleIndex) {
    var meta = graph._fullLayout.meta;
    var seconds = sampleIndex * meta.time_step_s;
    var epoch = new Date(meta.epoch_utc + (meta.epoch_utc.indexOf("Z") === -1 ? "Z" : ""));
    var now = new Date(epoch.getTime() + seconds * 1000);
    var stamp = now.getUTCFullYear() + "-" + pad(now.getUTCMonth() + 1) + "-" + pad(now.getUTCDate()) + " "
      + pad(now.getUTCHours()) + ":" + pad(now.getUTCMinutes());
    var span = document.getElementById("time-readout");
    if (span) {
      span.textContent = stamp + " UTC  (+" + (seconds / 86400).toFixed(3) + " d, "
        + (seconds / 375190.26).toFixed(4) + " TU)";
    }
  }

  function frame(timestamp) {
    if (!state.playing) { return; }
    var shown = graphs();
    if (shown.length > 0) {
      var meta = shown[0]._fullLayout.meta;
      if (state.lastFrame !== null) {
        var elapsed = Math.min(0.5, (timestamp - state.lastFrame) / 1000.0);
        state.index = (state.index + elapsed * state.hoursPerSecond * 3600.0 / meta.time_step_s) % meta.n_samples;
      }
      state.frameCount += 1;
      var includeBodies = state.frameCount % 4 === 0;
      shown.forEach(function (graph) { draw(graph, state.index, includeBodies); });
      readout(shown[0], state.index);
    }
    state.lastFrame = timestamp;
    state.handle = window.requestAnimationFrame(frame);
  }

  // Exposed for testing from the console: window.cislunarPlayback.draw(graph, sampleIndex, true).
  window.cislunarPlayback = {draw: draw, state: state, graphs: graphs};

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    playback: {
      control: function (playing, hoursPerSecond, sliderValue) {
        state.hoursPerSecond = parseFloat(hoursPerSecond) || 6.0;
        if (playing && !state.playing) {
          state.playing = true;
          state.index = sliderValue || 0;
          state.lastFrame = null;
          state.frameCount = 0;
          state.handle = window.requestAnimationFrame(frame);
          return window.dash_clientside.no_update;
        }
        if (!playing && state.playing) {
          state.playing = false;
          if (state.handle) { window.cancelAnimationFrame(state.handle); }
          return Math.round(state.index);
        }
        return window.dash_clientside.no_update;
      }
    }
  });
})();

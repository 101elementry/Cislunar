/* Flat projections of a few orbits, for scenes where the point is the
   shape seen from a chosen side rather than a 3D view (the NRHO page).

   app/showcase.py embeds, for each orbit, one period sampled evenly in
   time in the Moon-centred rotating frame (km), with the distance from
   the Moon's centre and the rotating-frame speed at each sample.  This
   script only draws: it picks two of the three coordinates, scales them
   to the canvas and moves a marker from sample to sample, so the marker's
   speed on screen is the spacecraft's speed. */

(function () {
  var page = JSON.parse(document.getElementById("page-data").textContent);
  var canvas = document.getElementById("orbit-canvas");
  var context = canvas.getContext("2d");
  var playButton = document.getElementById("play-button");
  var scrubber = document.getElementById("scrubber");
  var readout = document.getElementById("time-readout");
  var caption = document.getElementById("view-caption");

  // Which coordinate runs across and which runs up, and the words at the
  // edges of the drawing.  The x-axis points from the Earth to the Moon,
  // z is north, and y is the direction the Moon moves along its orbit.
  var VIEWS = {
    xz: {across: "x", up: "z", caption: "side-on, looking along the Moon's direction of travel",
         left: "towards Earth", right: "away from Earth", top: "north", bottom: "south"},
    yz: {across: "y", up: "z", caption: "from Earth, looking along the Earth-Moon line",
         left: "", right: "direction of the Moon's travel", top: "north", bottom: "south"},
    xy: {across: "x", up: "y", caption: "from above the north pole",
         left: "towards Earth", right: "away from Earth", top: "direction of the Moon's travel", bottom: ""}
  };

  var view = "xz";
  var orbit = page.order[0];
  var sample = 0;
  var playing = false;
  var samplesPerSecond = 40;

  function cssColour(name, fallback) {
    var value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  // One scale for every orbit in a view, so switching orbits keeps the
  // same scale and the orbits can be compared by eye.
  function layout(width, height) {
    var v = VIEWS[view];
    var lowAcross = -2000, highAcross = 2000, lowUp = -2000, highUp = 2000;
    page.order.forEach(function (key) {
      var o = page.orbits[key];
      o[v.across].forEach(function (value) { lowAcross = Math.min(lowAcross, value); highAcross = Math.max(highAcross, value); });
      o[v.up].forEach(function (value) { lowUp = Math.min(lowUp, value); highUp = Math.max(highUp, value); });
    });
    var margin = 48;
    var scale = Math.min((width - 2 * margin) / (highAcross - lowAcross), (height - 2 * margin) / (highUp - lowUp));
    var centreAcross = width / 2 - scale * (lowAcross + highAcross) / 2;
    var centreUp = height / 2 + scale * (lowUp + highUp) / 2;
    return {scale: scale, pixel: function (a, u) { return [centreAcross + scale * a, centreUp - scale * u]; }};
  }

  function draw() {
    var ratio = window.devicePixelRatio || 1;
    var width = canvas.clientWidth, height = canvas.clientHeight;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    var v = VIEWS[view];
    var frame = layout(width, height);
    var muted = cssColour("--text-muted", "#6e6e6e");
    var rule = cssColour("--rule-strong", "#303030");

    // The Earth-Moon line (or, from Earth, the Moon's orbital plane) through the Moon.
    var origin = frame.pixel(0, 0);
    context.strokeStyle = rule;
    context.lineWidth = 1;
    context.setLineDash([4, 4]);
    context.beginPath();
    context.moveTo(0, origin[1]);
    context.lineTo(width, origin[1]);
    context.stroke();
    context.setLineDash([]);

    // Every orbit: the chosen one in the trajectory colour, the others as context.
    page.order.forEach(function (key) {
      var o = page.orbits[key];
      context.beginPath();
      for (var k = 0; k < o.x.length; k++) {
        var p = frame.pixel(o[v.across][k], o[v.up][k]);
        if (k === 0) { context.moveTo(p[0], p[1]); } else { context.lineTo(p[0], p[1]); }
      }
      context.strokeStyle = key === orbit ? page.highlight : rule;
      context.lineWidth = key === orbit ? 2 : 1;
      context.stroke();
    });

    // The Moon, to scale.
    context.fillStyle = page.moon_color;
    context.beginPath();
    context.arc(origin[0], origin[1], Math.max(2.5, page.moon_radius_km * frame.scale), 0, 2 * Math.PI);
    context.fill();

    // The spacecraft.
    var o = page.orbits[orbit];
    var here = frame.pixel(o[v.across][sample], o[v.up][sample]);
    context.fillStyle = "#ffffff";
    context.fillRect(here[0] - 3.5, here[1] - 3.5, 7, 7);

    // Directions at the edges.
    context.fillStyle = muted;
    context.font = "500 10.5px Inter, -apple-system, sans-serif";
    context.textBaseline = "middle";
    context.textAlign = "left";
    // The bottom 40 px belong to the view caption, so the lower labels sit above it.
    if (v.left) { context.fillText("← " + v.left.toUpperCase(), 16, height - 52); }
    if (v.top) { context.fillText("↑ " + v.top.toUpperCase(), 16, 20); }
    if (v.bottom) { context.fillText("↓ " + v.bottom.toUpperCase(), 16, height - 76); }
    context.textAlign = "right";
    if (v.right) { context.fillText(v.right.toUpperCase() + " →", width - 16, height - 52); }
    context.fillText("MOON TO SCALE", width - 16, 20);

    caption.textContent = v.caption;
    var days = o.period_days * sample / (o.x.length - 1);
    readout.textContent = "day " + days.toFixed(2) + " of " + o.period_days.toFixed(2) + "   " +
      Math.round(o.distance_km[sample]).toLocaleString("en-AU") + " km from the Moon's centre   " +
      o.speed_km_s[sample].toFixed(2) + " km/s";
    scrubber.value = sample;
  }

  function markChoices() {
    document.querySelectorAll("[data-view]").forEach(function (button) {
      button.classList.toggle("playing", button.dataset.view === view);
    });
    document.querySelectorAll("[data-orbit]").forEach(function (button) {
      button.classList.toggle("playing", button.dataset.orbit === orbit);
    });
  }

  function setPlaying(next) {
    playing = next;
    playButton.textContent = playing ? "Pause" : "Play";
    playButton.classList.toggle("playing", playing);
  }

  // Advance whole samples at a fixed rate, so equal times take equal
  // screen time and the marker's speed is the spacecraft's speed.
  var last = null, carried = 0;
  function tick(now) {
    if (playing) {
      if (last !== null) {
        carried += (now - last) / 1000 * samplesPerSecond;
        var steps = Math.floor(carried);
        if (steps > 0) {
          carried -= steps;
          sample = (sample + steps) % (page.orbits[orbit].x.length - 1);
          draw();
        }
      }
      last = now;
    } else {
      last = null;
    }
    window.requestAnimationFrame(tick);
  }

  document.querySelectorAll("[data-view]").forEach(function (button) {
    button.addEventListener("click", function () { view = button.dataset.view; markChoices(); draw(); });
  });
  document.querySelectorAll("[data-orbit]").forEach(function (button) {
    button.addEventListener("click", function () { orbit = button.dataset.orbit; sample = 0; markChoices(); draw(); });
  });
  playButton.addEventListener("click", function () { setPlaying(!playing); });
  scrubber.addEventListener("input", function () { sample = parseInt(scrubber.value, 10); draw(); });
  document.addEventListener("keydown", function (event) {
    if (event.code === "Space" && event.target === document.body) { event.preventDefault(); setPlaying(!playing); }
  });
  window.addEventListener("resize", draw);

  markChoices();
  draw();
  var still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!still) { setPlaying(true); }
  window.requestAnimationFrame(tick);
})();

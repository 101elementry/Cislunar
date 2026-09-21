/* Static showcase page: draws the precomputed figures and drives the
   playback clock without Dash.

   The figures and the run's clock are embedded in the page as JSON by
   app/showcase.py.  The motion itself is done by playback.js, the same
   script the interface uses; this file only supplies what Dash supplies
   there: the play button, the speed choice and the time slider.
   playback.js finds its graphs by the holder ids view-3d and view-3d-b
   and writes the clock text into #time-readout. */

(function () {
  var page = JSON.parse(document.getElementById("page-data").textContent);
  var holders = ["view-3d", "view-3d-b"];
  // Plotly's toolbar sits over the legend on a phone, where pinch and drag do its job anyway.
  var config = {displaylogo: false, responsive: true, displayModeBar: window.innerWidth > 860 ? "hover" : false};

  var playButton = document.getElementById("play-button");
  var speedSelect = document.getElementById("play-speed");
  var scrubber = document.getElementById("scrubber");
  var accessLight = document.getElementById("access-light");
  var playing = false;

  function control(play, sliderValue) {
    return window.dash_clientside.playback.control(play, parseFloat(speedSelect.value), sliderValue);
  }

  // True if the time of this sample falls inside an access window.
  function inAccess(sampleIndex) {
    if (!page.windows_s) { return false; }
    var seconds = sampleIndex * page.time_step_s;
    for (var k = 0; k < page.windows_s.length; k++) {
      if (seconds >= page.windows_s[k][0] && seconds <= page.windows_s[k][1]) { return true; }
    }
    return false;
  }

  function showAccess(sampleIndex) {
    if (!accessLight) { return; }
    var on = inAccess(sampleIndex);
    accessLight.classList.toggle("on", on);
    accessLight.textContent = on ? "Sydney has access" : "no access";
  }

  // Draw every scene at one sample while playback is stopped.
  function showSample(sampleIndex) {
    var graphs = window.cislunarPlayback.graphs();
    graphs.forEach(function (graph) { window.cislunarPlayback.draw(graph, sampleIndex, true); });
    if (graphs.length > 0) { window.cislunarPlayback.readout(graphs[0], sampleIndex); }
    showAccess(sampleIndex);
  }

  function setPlaying(next) {
    if (next === playing) { return; }
    playing = next;
    playButton.textContent = playing ? "Pause" : "Play";
    playButton.classList.toggle("playing", playing);
    if (playing) {
      control(true, parseInt(scrubber.value, 10));
    } else {
      var reached = control(false, 0);
      if (typeof reached === "number") { scrubber.value = reached; showSample(reached); }
    }
  }

  // While playing, keep the slider and the access light in step with
  // the clock that playback.js is running.
  function follow() {
    if (playing) {
      var index = window.cislunarPlayback.state.index;
      scrubber.value = Math.round(index);
      showAccess(index);
    }
    window.requestAnimationFrame(follow);
  }

  playButton.addEventListener("click", function () { setPlaying(!playing); });
  speedSelect.addEventListener("change", function () { if (playing) { control(true, 0); } });
  scrubber.addEventListener("input", function () {
    var index = parseInt(scrubber.value, 10);
    if (playing) { window.cislunarPlayback.state.index = index; } else { showSample(index); }
  });
  document.addEventListener("keydown", function (event) {
    if (event.code === "Space" && event.target === document.body) { event.preventDefault(); setPlaying(!playing); }
  });

  var drawn = page.figures.map(function (figure, k) {
    var plot = document.getElementById(holders[k]).querySelector(".plot");
    return Plotly.newPlot(plot, figure.data, figure.layout, config);
  });
  if (page.series) {
    drawn.push(Plotly.newPlot(document.getElementById("series"), page.series.data, page.series.layout,
                              {displayModeBar: false, responsive: true}));
  }

  Promise.all(drawn).then(function () {
    showSample(0);
    window.requestAnimationFrame(follow);
    var still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!still) { setPlaying(true); }
  });
})();

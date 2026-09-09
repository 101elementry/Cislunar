/* Zoom toward the point under the cursor in the 3D scene.

   Plotly's own scroll zoom moves the camera toward its centre point,
   which is rarely where you are looking.  This handler runs first: if
   the cursor is over a trace (an orbit, a body, a marker), the camera
   is moved so that the hovered point stays fixed on screen while the
   eye approaches it; over empty space nothing is intercepted and
   Plotly zooms as usual.

   Camera coordinates: Plotly maps the data box so each axis spans
   [-aspectratio / 2, +aspectratio / 2] around the origin (checked from
   the scene's model matrix), so a data point at fraction f along an
   axis sits at (f - 1/2) * aspectratio. */

(function () {
  function attach(graph) {
    if (graph.dataset.zoomToCursor) { return; }
    graph.dataset.zoomToCursor = "1";
    var hovered = null;
    graph.on("plotly_hover", function (event) {
      var point = event.points && event.points[0];
      hovered = point && typeof point.x === "number" ? [point.x, point.y, point.z] : null;
    });
    graph.on("plotly_unhover", function () { hovered = null; });

    graph.addEventListener("wheel", function (event) {
      if (!hovered || !graph._fullLayout || !graph._fullLayout.scene) { return; }
      var scene = graph._fullLayout.scene;
      var camera = scene.camera;
      var axes = [scene.xaxis, scene.yaxis, scene.zaxis];
      var aspect = [scene.aspectratio.x, scene.aspectratio.y, scene.aspectratio.z];
      var target = [];
      for (var i = 0; i < 3; i++) {
        var range = axes[i].range;
        var fraction = (hovered[i] - range[0]) / (range[1] - range[0]);
        target.push((fraction - 0.5) * aspect[i]);
      }
      // Scale the eye-to-target vector: shrinking it moves in, keeping
      // the target fixed on screen.  Trackpads send small deltas, wheels
      // large ones; the exponent keeps both smooth.
      var factor = Math.exp(event.deltaY * 0.002);
      var eye = [camera.eye.x, camera.eye.y, camera.eye.z];
      var newEye = [];
      for (var k = 0; k < 3; k++) {
        newEye.push(target[k] + (eye[k] - target[k]) * factor);
      }
      // Move the centre toward the target as well so turning afterwards
      // orbits what you zoomed into.
      var centre = [camera.center.x, camera.center.y, camera.center.z];
      var newCentre = [];
      for (var m = 0; m < 3; m++) {
        newCentre.push(centre[m] + (target[m] - centre[m]) * 0.35);
      }
      event.preventDefault();
      event.stopImmediatePropagation();
      Plotly.relayout(graph, {
        "scene.camera.eye": {x: newEye[0], y: newEye[1], z: newEye[2]},
        "scene.camera.center": {x: newCentre[0], y: newCentre[1], z: newCentre[2]}
      });
    }, {capture: true, passive: false});
  }

  function look() {
    var graph = document.querySelector("#view-3d .js-plotly-plot");
    if (graph && graph.on) { attach(graph); }
  }
  var observer = new MutationObserver(look);
  observer.observe(document.documentElement, {childList: true, subtree: true});
  look();
})();

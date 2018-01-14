// Code for visualizing the clustering derived from
// https://bl.ocks.org/pbogden/854425acb57b4e5a4fdf4242c068a127
var NODE_RADIUS = Math.sqrt(2);
var CENTROID_RADIUS = Math.sqrt(10);

Array.prototype.unique = function() {
  return this.filter(function (value, index, self) { 
    return self.indexOf(value) === index;
  });
}

// https://tc39.github.io/ecma262/#sec-array.prototype.findIndex
if (!Array.prototype.findIndex) {
  Object.defineProperty(Array.prototype, 'findIndex', {
    value: function(predicate) {
     // 1. Let O be ? ToObject(this value).
      if (this == null) {
        throw new TypeError('"this" is null or not defined');
      }

      var o = Object(this);

      // 2. Let len be ? ToLength(? Get(O, "length")).
      var len = o.length >>> 0;

      // 3. If IsCallable(predicate) is false, throw a TypeError exception.
      if (typeof predicate !== 'function') {
        throw new TypeError('predicate must be a function');
      }

      // 4. If thisArg was supplied, let T be thisArg; else let T be undefined.
      var thisArg = arguments[1];

      // 5. Let k be 0.
      var k = 0;

      // 6. Repeat, while k < len
      while (k < len) {
        // a. Let Pk be ! ToString(k).
        // b. Let kValue be ? Get(O, Pk).
        // c. Let testResult be ToBoolean(? Call(predicate, T, « kValue, k, O »)).
        // d. If testResult is true, return k.
        var kValue = o[k];
        if (predicate.call(thisArg, kValue, k, o)) {
          return k;
        }
        // e. Increase k by 1.
        k++;
      }

      // 7. Return -1.
      return -1;
    }
  });
}
        
function displayClustering(displayElementID, nodes) {
    if (nodes.length === 0) return;
    
     // Read the size from the element enclosing the SVG element.
     var PADDING = 10;
     var displayElement = document.getElementById(displayElementID);
     var width = displayElement.offsetWidth - (PADDING * 2);
     var height = window.innerHeight - displayElement.offsetTop - (PADDING * 2);
     var max_x = width/2;
     var min_x = -max_x;
     var max_y = height/2;
     var min_y = -max_y;

    // Create an array of nodes to use as the centroids for each cluster.
    // Make the radius for the corresponding node larger.
    var centroids = new Array(1);
    nodes[0].radius = CENTROID_RADIUS;
    centroids[0] = nodes[0];
    centroids[0].centroidIndex = 0;
    for (i = 1; i < nodes.length; i++) {
        if (centroids.findIndex( function(d) { return d.cluster === nodes[i].cluster } ) >= 0) {
            // This cluster already has a centroid, so use the node radius.
            nodes[i].radius = NODE_RADIUS;
        } else {
            // First node for this cluster; use it as the centroid.
            nodes[i].radius = CENTROID_RADIUS;
            new_centroid = nodes[i];
            new_centroid.centroidIndex = i;
            centroids.push(new_centroid)
        }
    }

    // Initialize the cluster colors to be one of the schemeCategory20
    // colors, reusing colors if there are more than 20 clusters.
    var color = new Array(centroids.length);
    for (i=0; i < color.length; i++) {
        color[i] = d3.schemeCategory20[i % 19];
    }

    function centroidRadius(d) {
        return d.radius + (NODE_RADIUS * 1.5);
    }
    
    var forceCollide = d3.forceCollide()
        .radius(centroidRadius)
        .iterations(1);

    // Update the circle centers, while keeping the circles within
    // the display area.
    function tick() {
      circle
          .attr("cx", function(d) { 
              if (d.x < min_x + d.radius) {
                  d.x = min_x + d.radius;
              } else if (d.x > max_x - d.radius) {
                  d.x = max_x - d.radius;
              }
              return d.x; 
          })
          .attr("cy", function(d) {
               if (d.y < min_y + d.radius) {
                   d.y = min_y + d.radius;
               } else if (d.y > max_y - d.radius) {
                   d.y = max_y - d.radius;
               }
               return d.y;
          });
    }

    // Cause the nodes to cluster around their associated centroid.
    function forceCluster(alpha) {
      for (i = 0; i < nodes.length; ++i) {
        var j = centroids.findIndex(function(d) { return d.cluster === nodes[i].cluster });
        nodes[i].vx -= (nodes[i].x - nodes[centroids[j].centroidIndex].x) * alpha;
        nodes[i].vy -= (nodes[i].y - nodes[centroids[j].centroidIndex].y) * alpha;
      }
    }
    
    var force = d3.forceSimulation()
        .nodes(nodes)
        .force("center", d3.forceCenter())
        .force("collide", forceCollide)
        .force("cluster", forceCluster)
        .force("gravity", d3.forceManyBody(30))
        .force("x", d3.forceX().strength(.7))
        .force("y", d3.forceY().strength(.7))
        .on("tick", tick);


    var displayElementSelection = d3.select("#" + displayElementID);
    var svg = displayElementSelection.select("svg");
    if (svg.empty()) {
        svg = displayElementSelection.append("svg")
               .attr("width", width)
               .attr("height", height)
                 .append('g')
                   .attr('transform', 'translate(' + width / 2 + ',' + height / 2 + ')');
    }

    var circle = svg.selectAll("circle")
        .data(nodes)
          .enter().append("circle")
        .attr("r", function(d) { return d.radius; })
        .style("fill", function(d) { return color[centroids.findIndex( function(e) { return d.cluster === e.cluster } )]; })

}

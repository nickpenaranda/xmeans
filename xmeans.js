var xmeans = {}

// Generate uniformly random data
xmeans.genTestData = function(dims, n) {
    var data = [];
    for (var i = 0; i < n; i++) {
        var row = [];
        for (var j = 0; j < dims; j++) {
            // Don't change this interval or else rendering will fail
            row.push(xmeans._random(0,500));
        }
        data.push(row);
    }

    return data;
}

// Generate clustered data
xmeans.genTestData2 = function(dims, n, k, sd) {
    var data = [];
    var generators = [];

    for (var i = 0; i < k; i++) {
        var centroid = [];
        for (var j = 0; j < dims; j++) {
            centroid.push(xmeans._random(100,900));
        }

        generators.push(function() {
            return this.map(function(e) {
                return xmeans._randomN(e, sd);
            });
        }.bind(centroid));
    }

    for (var i = 0; i < n; i++) {
        data.push(generators[i % k]());
    }

    return data;
}

// Generate 2d clustered data on a diagonal
xmeans.genTestData3 = function() {
    var data = [];
    for (var i = 0; i < 4; i++) {
        for (var j = 0; j < 200; j++) {
            data.push([
                xmeans._randomN(100+(100*i),10),
                xmeans._randomN(100+(100*i),10)
            ]);
        }
    }
    return data;
}

xmeans._extractVec = function(x, dataFields) {
    if (dataFields == null) {
        return x;
    }

    var nDim = dataFields.length;
    var vec = Array(nDim);
    for (var i=0; i < nDim; i++) {
        vec[i] = x[dataFields[i]];
    }

    return vec;
}

// Shallow merge
xmeans._mergeOptions = function(a, b) {
    var o = {};
    for (var attr in a) {
        o[attr] = a[attr];
    }

    for (var attr in b) {
        o[attr] = b[attr];
    }

    return o;
}

xmeans._randomNormalVec = function(dims) {
    var vec = [];
    for (var i = 0; i < dims; i++) {
        vec.push(-1 + (Math.random() * 2));
    }

    return xmeans._normVec(vec);
}

xmeans._normVec = function(vec) {
    var veclen = xmeans._magnitude(vec);

    return vec.map(function(e) {
        return e / veclen;
    });
}

xmeans._addVec = function(x,y) {
    return x.map(function(e,i) {
        return e + y[i];
    });
}

xmeans._scaleVec = function(x,sc) {
    return x.map(function(e) {
        return e * sc;
    });
}

xmeans._invVec = function(x) {
    return xmeans._scaleVec(x,-1);
}

xmeans._center = function(data, fields) {
    if (data.length == 0 || data[0].length == 0) {
        throw "Invalid data";
    }

    var mustExtract = fields != null;
    var n = data.length;
    var m = mustExtract ? fields.length : data[0].length;

    var result = Array.apply(null, Array(m)).map(Number.prototype.valueOf, 0);

    var sum = data.reduce(function(memo, i) {
        var p = mustExtract ? xmeans._extractVec(i, fields) : i;
        p.forEach(function(j,k) {
            memo[k] += j;
        });
        return memo;
    }, result);

    return sum.map(function(e) {
        return e/n;
    });
}

xmeans._magnitude = function(x) {
    return Math.sqrt(x.reduce(function(memo, e) {
        return memo + Math.pow(e,2);
    }, 0));
}

xmeans._distance = function(x,y) {
    if (x.length != y.length) {
        throw "Invalid data";
    }

    var len = x.length;
    var result = Array(len);
    for (var i=0; i < len; i++) {
        result[i] = y[i] - x[i];
    }

    return xmeans._magnitude(result);
}

// Returns [lowerBound, upperBound], both vectors of same dimension as vecs
xmeans._bounds = function(vecs, fields) {
    if (vecs.length == 0 || vecs[0].length == 0) {
        throw "Invalid data";
    }

    var m = [
        xmeans._extractVec(vecs[0], fields),
        xmeans._extractVec(vecs[0], fields)
    ];

    if (fields == null) {
        fields = [];
        for (var i = 0; i < vecs[0].length; i++) {
            fields.push(i);
        }
    }

    return vecs.reduce(function(memo,i) {
        fields.forEach(function(j,k) {
            if (i[j] < memo[0][k]) {
                memo[0][k] = i[j];
            } else if (i[j] > memo[1][k]) {
                memo[1][k] = i[j];
            }
        });

        return memo;
    },m);
}

xmeans._random = function(min,max) {
    return Math.random() * (max-min) + min;
}

xmeans._randomN = function(mean,stdev) {
    var rnd = (Math.random() + Math.random() + Math.random() +
               Math.random() + Math.random() + Math.random() - 3) * 1.41421356237;
    return mean + (rnd * stdev);
}

// Returns array of k clusters, which are objects with fields:
// centroid: vector
// size: radius of bounding circle
// points: array of vectors
// sumSqDist: array of squared _distances for each point
xmeans._kmeans = function(data, options) {
    if (options != null) {
        if (options.k == undefined) {
            throw "Missing required option: k"
        }
    } else {
        throw "Missing required parameter: option (object)"
    }

    var dims = options.dataFields == undefined
        ? data[0].length
        : options.dataFields.length;

    var b = xmeans._bounds(data, options.dataFields);
    var n = data.length;

    var clusters = [];

    if (options.startCentroids == undefined) {
        for (var i = 0; i < options.k; i++) {
            var centroid = [];
            for (var j = 0; j < dims; j++) {
                centroid.push(xmeans._random(b[0][j],b[1][j]));
            }
            clusters.push({centroid: centroid});
        }
    } else {
        for (var i = 0; i < options.k; i++) {
            clusters.push({centroid: options.startCentroids[i]});
        }
    }

    var nIters = 0;
    while (nIters < 100) {
        nIters++;

        // Clear centroid point sets
        for (var i = 0; i < options.k; i++) {
            clusters[i].points = [];
        }

        // Assign
        for (var i = 0; i < n; i++) {
            var point = xmeans._extractVec(data[i], options.dataFields);
            var closestClusterIndex = 0;
            var closestClusterDist = Infinity;
            for (var j = 0; j < options.k; j++) {
                var centroid = clusters[j].centroid;
                var dist = xmeans._distance(centroid, point);
                if (dist < closestClusterDist) {
                    closestClusterDist = dist;
                    closestClusterIndex = j;
                }
            }
            clusters[closestClusterIndex].points.push(data[i]);
        }

        // Adjust centroids
        var distMoved = 0;
        for (var i = 0; i < options.k; i++) {
            if (clusters[i].points.length == 0) {
                continue;
            }

            var newCentroid = xmeans._center(clusters[i].points, options.dataFields);
            var moveDistance = xmeans._distance(newCentroid, clusters[i].centroid);
            clusters[i].centroid = newCentroid;
            distMoved += moveDistance;
        }

        if (distMoved == 0) {
            break;
        }
    }

    // Calculate cell sizes and store squared distances (for xmeans)
    for (var i = 0; i < options.k; i++) {
        var cluster = clusters[i];
        cluster.sumSqDist = 0;
        cluster.size = cluster.points.reduce(function(memo,p) {
            var dist = xmeans._distance(cluster.centroid, xmeans._extractVec(p, options.dataFields));
            cluster.sumSqDist += Math.pow(dist, 2);
            if (dist > memo) {
                return dist;
            } else {
                return memo;
            }
        }, 0);
    }

    return clusters;
}

xmeans._countPoints = function(clusters) {
    return clusters.reduce(function(memo, c) {
        return memo + c.points.length;
    }, 0);
}

xmeans._freeParams = function(k, nDims) {
    return k * (nDims + 1)
    // Explicit form:
    // return k - 1 + (nDims * k) + 1;
}

xmeans._variance = function(nPoints, nDims, clusters) {
    var s = clusters.reduce(function(memo1, c) {
        return memo1 + c.sumSqDist;
    }, 0);

    var denom = nDims * (nPoints - clusters.length);
    return s / denom;
}

xmeans.bic = function(nPoints, nDims, clusters, label) {
    var K = clusters.length;
    var R = nPoints;
    var M = nDims;
    var variance = xmeans._variance(nPoints, nDims, clusters);
    var p = xmeans._freeParams(K, nDims);
    var paramTerm = (p / 2) * Math.log(R);
    if (label == undefined) {
        label = ""
    }

    return clusters.reduce(function(memo,c) {
        var Rn = c.points.length;
        var t1 = Rn * Math.log(Rn);
        var t2 = (M * K) / 2;
        var t3 = ((Rn * M) / 2) * Math.log(variance);
        return memo + t1 - t2 - t3;
    },0) - paramTerm;
}

// Returns k and an array of clusters
xmeans.xmeans = function(data, options) {
    // TODO: DRY
    if (options != null) {
        // Individual defaults
        if (options.kMin == undefined) {
            options.kMin = 2;
        }

        if (options.kMax == undefined) {
            options.kMax = Math.floor(data.length / 4);
        }
    } else {
        // All defaults
        options = {};
        options.kMin = 2;
        options.kMax = Math.floor(data.length / 4);
    }

    var k = options.kMin;
    var nDims = options.dataFields != null
            ? options.dataFields.length
            : data[0].length;
    var nGlobalPoints = data.length;

    var clusters = xmeans._kmeans(data, xmeans._mergeOptions(options, {k: k}));

    while (true) {
        var startCentroids = clusters.map(function(c) {
            return c.centroid;
        });
        k = clusters.length;

        // Improve-Structure
        var oldGlobalBIC = xmeans.bic(nGlobalPoints, nDims, clusters);

        // Improve-Params
        clusters = xmeans._kmeans(data, xmeans._mergeOptions(options,{k: k, startCentroids: startCentroids}));

        for (var i = 0; i < k; i++) {
            var cluster = clusters[i];
            if (cluster.points.length == 0) {
                clusters.splice(i,1);
                i--;
                k--;
                continue;
            }

            ////// Split each centroid

            // Find split vector
            // TODO: What should scaling factor be? (set to 0.5)
            var baseVec = xmeans._randomNormalVec(nDims);
            var splitVector = xmeans._scaleVec(baseVec, cluster.size * 0.5);

            // Calculate new centroids
            var newCentroids = [
                xmeans._addVec(cluster.centroid, splitVector),
                xmeans._addVec(cluster.centroid, xmeans._invVec(splitVector))
            ];

            // Run local k-means on cluster with child centroids
            var newClusters = xmeans._kmeans(cluster.points, xmeans._mergeOptions(options, {k:2, startCentroids:newCentroids}));

            var nPoints = cluster.points.length;
            var oldBIC = xmeans.bic(nPoints, nDims, [cluster]);
            var newBIC = xmeans.bic(nPoints, nDims, newClusters);

            if (newBIC > oldBIC) {
                Array.prototype.splice.apply(clusters, [i,1].concat(newClusters));
                i++;
                k++;
            }
        }

        var newGlobalBIC = xmeans.bic(nGlobalPoints, nDims, clusters);
        if (newGlobalBIC <= oldGlobalBIC || k >= options.kMax) {
            break;
        }
    }

    return clusters;
}

xmeans.oxmeans = function(data, options) {
    if (options != null) {
        if (options.iters == undefined) {
            options.iters = 3;
        }
    } else {
        options = {};
        options.iters = 3;
    }

    var bestBIC = -Infinity;
    var bestModel = null;

    for (var i=0; i < options.iters; i++) {
        var result = xmeans.xmeans(data, options);
        var resultBIC = xmeans.bic(data.length, data[0].length, result);
        if (resultBIC > bestBIC) {
            bestBIC = resultBIC;
            bestModel = result;
        }
    }

    return bestModel;
}

xmeans.colors = ["red","green","blue","black","darkorange","blueviolet","deeppink","darkolivegreen"];

xmeans.zVector = [Math.sqrt(2) / 2, Math.sqrt(2) / 3];

xmeans._renderCircle = function(ctx,x,y,z,radius) {
    if (z == null) {
        z = 0;
    }

    ctx.arc(x + (z*xmeans.zVector[0]), 1000 - (y + (z*xmeans.zVector[1])), radius, 0, Math.PI * 2);
}

xmeans.render = function(clusters) {
    var ctx = document.getElementById("canvas").getContext("2d");
    ctx.clearRect(0,0,1000,1000);

    var is3d = clusters[0].centroid.length > 2;

    if (is3d) {
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = "black";

        ctx.beginPath();
        ctx.rect(0,500,500,500);
        ctx.stroke();

        ctx.rect(xmeans.zVector[0]*500,500-xmeans.zVector[1]*500,500,500);
        ctx.stroke();

        var corners = [
            [0, 500, xmeans.zVector[0]*500, 500-xmeans.zVector[1]*500],
            [0, 1000, xmeans.zVector[0]*500, 1000-xmeans.zVector[1]*500],
            [500, 500, 500+xmeans.zVector[0]*500, 500-xmeans.zVector[1]*500],
            [500, 1000, 500+xmeans.zVector[0]*500, 1000-xmeans.zVector[1]*500],
        ];

        corners.forEach(function(l) {
            ctx.beginPath();
            ctx.moveTo(l[0],l[1]);
            ctx.lineTo(l[2],l[3]);
            ctx.stroke();
        });
    }

    var nColors = xmeans.colors.length;
    clusters.forEach(function(e,i) {
        ctx.fillStyle = xmeans.colors[i % nColors];
        ctx.strokeStyle = xmeans.colors[i % nColors];
        ctx.setLineDash([]);

        // Points
        e.points.forEach(function(p) {
            ctx.beginPath();
            xmeans._renderCircle(ctx,p[0],p[1],is3d ? p[2] : 0,2);
            ctx.fill();
        });

        // Centroid
        ctx.beginPath();
        xmeans._renderCircle(ctx,e.centroid[0],e.centroid[1],e.centroid[2],4);
        ctx.stroke();

        // Bounding circle
        ctx.setLineDash([4,8]);
        ctx.beginPath();
        xmeans._renderCircle(ctx,e.centroid[0],e.centroid[1],e.centroid[2],e.size);
        ctx.stroke();
    });
}



/**
 * memory3d.js – Three.js 3D Memory Visualization
 * Solid particles with kNN connection lines, orbit controls, hover tooltips
 */

const Memory3D = (() => {
    "use strict";

    let scene, camera, renderer, controls;
    let particles, particleData = [];
    let raycaster, mouse;
    let tooltip;
    let container;
    let initialized = false;
    let animId = null;
    let connectionLines = null;

    // ── Solid dot texture ──
    function createDotTexture() {
        const canvas = document.createElement("canvas");
        canvas.width = 64;
        canvas.height = 64;
        const ctx = canvas.getContext("2d");
        const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
        gradient.addColorStop(0, "rgba(255,255,255,1)");
        gradient.addColorStop(0.3, "rgba(255,255,255,0.95)");
        gradient.addColorStop(0.5, "rgba(180,220,255,0.6)");
        gradient.addColorStop(0.7, "rgba(100,160,255,0.2)");
        gradient.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 64, 64);
        return new THREE.CanvasTexture(canvas);
    }

    // ── Color by age ──
    function getColor(timestamp, minTs, maxTs) {
        if (maxTs === minTs) return new THREE.Color(0x66ccff);
        const t = (timestamp - minTs) / (maxTs - minTs);
        const r = 0.1 + 0.2 * (1 - t);
        const g = 0.5 + 0.5 * t;
        const b = 0.9 - 0.5 * t;
        return new THREE.Color(r, g, b);
    }

    // ── Init ──
    function init(containerId) {
        container = document.getElementById(containerId);
        if (!container || initialized) return;
        requestAnimationFrame(() => _doInit());
    }

    function _doInit() {
        tooltip = document.getElementById("memoryTooltip");

        const w = container.clientWidth || 800;
        const h = container.clientHeight || 500;
        console.log("Memory3D init:", w, "x", h);

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a1a);
        scene.fog = new THREE.FogExp2(0x0a0a1a, 0.12);

        camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 100);
        camera.position.set(0, 0, 6);

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(w, h);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;
        controls.maxDistance = 15;
        controls.minDistance = 1;

        raycaster = new THREE.Raycaster();
        raycaster.params.Points.threshold = 0.15;
        mouse = new THREE.Vector2();

        const gridHelper = new THREE.GridHelper(10, 20, 0x1a1a3a, 0x1a1a3a);
        gridHelper.position.y = -3;
        scene.add(gridHelper);

        renderer.domElement.addEventListener("mousemove", onMouseMove);
        renderer.domElement.addEventListener("mouseleave", () => {
            if (tooltip) tooltip.style.opacity = "0";
        });
        window.addEventListener("resize", onResize);

        initialized = true;
        start();
    }

    // ── Mouse hover ──
    function onMouseMove(event) {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        if (particles) {
            const intersects = raycaster.intersectObject(particles);
            if (intersects.length > 0) {
                const idx = intersects[0].index;
                const point = particleData[idx];
                if (point && tooltip) {
                    tooltip.textContent = point.text;
                    const r2 = renderer.domElement.getBoundingClientRect();
                    tooltip.style.left = (event.clientX - r2.left + 15) + "px";
                    tooltip.style.top = (event.clientY - r2.top - 10) + "px";
                    tooltip.style.opacity = "1";
                }
                controls.autoRotate = false;
            } else {
                if (tooltip) tooltip.style.opacity = "0";
                controls.autoRotate = true;
            }
        }
    }

    function onResize() {
        if (!container || !camera || !renderer) return;
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    }

    // ── Data loading ──
    async function loadData() {
        try {
            const res = await fetch("/api/memory_3d");
            const json = await res.json();
            return json.points || [];
        } catch (e) {
            console.error("Error fetching memory 3D:", e);
            return [];
        }
    }

    // ── Build particles + connections ──
    function buildParticles(points) {
        if (particles) {
            scene.remove(particles);
            particles.geometry.dispose();
            particles.material.dispose();
        }
        if (connectionLines) {
            scene.remove(connectionLines);
            connectionLines.geometry.dispose();
            connectionLines.material.dispose();
            connectionLines = null;
        }

        if (points.length === 0) return;

        particleData = points;
        const count = points.length;

        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(count * 3);
        const colors = new Float32Array(count * 3);

        const timestamps = points.map(p => p.timestamp);
        const minTs = Math.min(...timestamps);
        const maxTs = Math.max(...timestamps);

        const posArr = [];

        for (let i = 0; i < count; i++) {
            const x = points[i].x * 3;
            const y = points[i].y * 3;
            const z = points[i].z * 3;
            positions[i * 3] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = z;
            posArr.push([x, y, z]);

            const color = getColor(points[i].timestamp, minTs, maxTs);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }

        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.35,
            map: createDotTexture(),
            vertexColors: true,
            transparent: true,
            opacity: 1.0,
            depthWrite: false,
            sizeAttenuation: true,
        });

        particles = new THREE.Points(geometry, material);
        scene.add(particles);

        buildConnections(posArr, colors, count);
    }

    // ── kNN connection lines ──
    function buildConnections(posArr, colors, count) {
        if (count < 2) return;

        const K = Math.min(2, count - 1);
        const linePositions = [];
        const lineColors = [];

        for (let i = 0; i < count; i++) {
            const dists = [];
            for (let j = 0; j < count; j++) {
                if (i === j) continue;
                const dx = posArr[i][0] - posArr[j][0];
                const dy = posArr[i][1] - posArr[j][1];
                const dz = posArr[i][2] - posArr[j][2];
                dists.push({ idx: j, d: dx * dx + dy * dy + dz * dz });
            }
            dists.sort((a, b) => a.d - b.d);

            for (let k = 0; k < K; k++) {
                const j = dists[k].idx;
                linePositions.push(posArr[i][0], posArr[i][1], posArr[i][2]);
                lineColors.push(colors[i * 3] * 0.4, colors[i * 3 + 1] * 0.4, colors[i * 3 + 2] * 0.4);
                linePositions.push(posArr[j][0], posArr[j][1], posArr[j][2]);
                lineColors.push(colors[j * 3] * 0.4, colors[j * 3 + 1] * 0.4, colors[j * 3 + 2] * 0.4);
            }
        }

        const lineGeometry = new THREE.BufferGeometry();
        lineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
        lineGeometry.setAttribute("color", new THREE.Float32BufferAttribute(lineColors, 3));

        const lineMaterial = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.35,
            linewidth: 1,
        });

        connectionLines = new THREE.LineSegments(lineGeometry, lineMaterial);
        scene.add(connectionLines);
    }

    // ── Animation loop ──
    function animate() {
        animId = requestAnimationFrame(animate);
        controls.update();

        if (particles) {
            const time = Date.now() * 0.0005;
            particles.rotation.y = Math.sin(time) * 0.05;
            if (connectionLines) connectionLines.rotation.y = particles.rotation.y;
        }

        renderer.render(scene, camera);
    }

    // ── Public API ──
    async function start() {
        if (!initialized) return;

        const countEl = document.getElementById("memoryCount");
        if (countEl) countEl.textContent = "Loading...";

        const points = await loadData();
        buildParticles(points);

        if (countEl) countEl.textContent = points.length + " memories";
        if (!animId) animate();
        onResize();
    }

    function stop() {
        if (animId) {
            cancelAnimationFrame(animId);
            animId = null;
        }
    }

    function destroy() {
        stop();
        if (renderer) {
            renderer.dispose();
            if (container && renderer.domElement.parentNode === container) {
                container.removeChild(renderer.domElement);
            }
        }
        initialized = false;
    }

    return { init, start, stop, destroy, onResize };
})();

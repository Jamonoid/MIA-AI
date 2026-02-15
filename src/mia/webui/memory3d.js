/**
 * memory3d.js – Three.js 3D Memory Visualization
 * Premium particle system with glow, orbit controls, hover tooltips
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

    // ── Glow texture (generated) ──
    function createGlowTexture() {
        const canvas = document.createElement("canvas");
        canvas.width = 64;
        canvas.height = 64;
        const ctx = canvas.getContext("2d");
        const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
        gradient.addColorStop(0, "rgba(255,255,255,1)");
        gradient.addColorStop(0.15, "rgba(120,200,255,0.8)");
        gradient.addColorStop(0.4, "rgba(60,120,255,0.3)");
        gradient.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 64, 64);
        return new THREE.CanvasTexture(canvas);
    }

    // ── Color by age ──
    function getColor(timestamp, minTs, maxTs) {
        if (maxTs === minTs) return new THREE.Color(0x66ccff);
        const t = (timestamp - minTs) / (maxTs - minTs); // 0=oldest, 1=newest
        // Blue → Cyan → Green gradient
        const r = 0.1 + 0.2 * (1 - t);
        const g = 0.5 + 0.5 * t;
        const b = 0.9 - 0.5 * t;
        return new THREE.Color(r, g, b);
    }

    function init(containerId) {
        container = document.getElementById(containerId);
        if (!container || initialized) return;

        // Tooltip
        tooltip = document.getElementById("memoryTooltip");

        // Scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a1a);
        scene.fog = new THREE.FogExp2(0x0a0a1a, 0.15);

        // Camera
        camera = new THREE.PerspectiveCamera(
            60, container.clientWidth / container.clientHeight, 0.1, 100
        );
        camera.position.set(0, 0, 5);

        // Renderer
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        // OrbitControls
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;
        controls.maxDistance = 15;
        controls.minDistance = 1;

        // Raycaster
        raycaster = new THREE.Raycaster();
        raycaster.params.Points.threshold = 0.15;
        mouse = new THREE.Vector2();

        // Ambient grid
        const gridHelper = new THREE.GridHelper(10, 20, 0x1a1a3a, 0x1a1a3a);
        gridHelper.position.y = -3;
        scene.add(gridHelper);

        // Events
        renderer.domElement.addEventListener("mousemove", onMouseMove);
        renderer.domElement.addEventListener("mouseleave", () => {
            if (tooltip) tooltip.style.opacity = "0";
        });
        window.addEventListener("resize", onResize);

        initialized = true;
    }

    function onMouseMove(event) {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        // Raycast
        raycaster.setFromCamera(mouse, camera);
        if (particles) {
            const intersects = raycaster.intersectObject(particles);
            if (intersects.length > 0) {
                const idx = intersects[0].index;
                const point = particleData[idx];
                if (point && tooltip) {
                    tooltip.textContent = point.text;
                    tooltip.style.left = (event.clientX - renderer.domElement.getBoundingClientRect().left + 15) + "px";
                    tooltip.style.top = (event.clientY - renderer.domElement.getBoundingClientRect().top - 10) + "px";
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

    function buildParticles(points) {
        // Remove old
        if (particles) {
            scene.remove(particles);
            particles.geometry.dispose();
            particles.material.dispose();
        }

        if (points.length === 0) return;

        particleData = points;
        const count = points.length;

        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(count * 3);
        const colors = new Float32Array(count * 3);
        const sizes = new Float32Array(count);

        const timestamps = points.map(p => p.timestamp);
        const minTs = Math.min(...timestamps);
        const maxTs = Math.max(...timestamps);

        for (let i = 0; i < count; i++) {
            // Scale coordinates for better spread
            positions[i * 3] = points[i].x * 3;
            positions[i * 3 + 1] = points[i].y * 3;
            positions[i * 3 + 2] = points[i].z * 3;

            const color = getColor(points[i].timestamp, minTs, maxTs);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;

            // Newer = slightly bigger
            const age = maxTs > minTs ? (points[i].timestamp - minTs) / (maxTs - minTs) : 0.5;
            sizes[i] = 0.15 + age * 0.15;
        }

        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));

        const material = new THREE.PointsMaterial({
            size: 0.3,
            map: createGlowTexture(),
            vertexColors: true,
            transparent: true,
            opacity: 0.9,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            sizeAttenuation: true,
        });

        particles = new THREE.Points(geometry, material);
        scene.add(particles);
    }

    function animate() {
        animId = requestAnimationFrame(animate);
        controls.update();

        // Gentle float animation
        if (particles) {
            const time = Date.now() * 0.0005;
            particles.rotation.y = Math.sin(time) * 0.05;
        }

        renderer.render(scene, camera);
    }

    async function start() {
        if (!initialized) return;

        const countEl = document.getElementById("memoryCount");
        if (countEl) countEl.textContent = "Loading...";

        const points = await loadData();
        buildParticles(points);

        if (countEl) {
            countEl.textContent = `${points.length} memories`;
        }

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

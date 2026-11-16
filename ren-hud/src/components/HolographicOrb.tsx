import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import type { HermesStatus } from "../types/gateway";

interface HolographicOrbProps {
  status: HermesStatus;
  rms: number;
  frequencies: number[];
}

const VERTEX_SHADER = /* glsl */ `
  uniform float uTime;
  uniform float uAmplitude;
  uniform float uHighFreq;

  varying vec3 vNormal;
  varying float vDisplacement;

  // Ruído procedural 3D (Simplex-like) para deformação orgânica de vértices
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

  float snoise(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);

    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;

    i = mod289(i);
    vec4 p = permute(permute(permute(
              i.z + vec4(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0));

    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);

    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);

    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);

    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
  }

  void main() {
    vNormal = normalize(normalMatrix * normal);
    float n1 = snoise(normal * (2.2 + uHighFreq * 2.5) + uTime * 1.4);
    float n2 = snoise(normal * 4.8 - uTime * 2.1) * 0.35;
    float disp = (n1 + n2) * (0.06 + uAmplitude * 0.42);
    vDisplacement = disp;

    vec3 newPosition = position + normal * disp;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
  }
`;

const FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uPrimaryColor;
  uniform vec3 uSecondaryColor;
  uniform float uAmplitude;

  varying vec3 vNormal;
  varying float vDisplacement;

  void main() {
    float fresnel = pow(1.0 - abs(dot(vNormal, vec3(0.0, 0.0, 1.0))), 1.8);
    float mixFactor = clamp(vDisplacement * 2.5 + 0.5 + fresnel * 0.4, 0.0, 1.0);
    vec3 color = mix(uPrimaryColor, uSecondaryColor, mixFactor);
    float alpha = clamp(0.55 + fresnel * 0.4 + uAmplitude * 0.25, 0.4, 0.98);
    gl_FragColor = vec4(color, alpha);
  }
`;

export const HolographicOrb: React.FC<HolographicOrbProps> = ({
  status,
  rms,
  frequencies,
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const targetRmsRef = useRef<number>(0);
  const targetHighFreqRef = useRef<number>(0);
  const statusRef = useRef<HermesStatus>(status);

  useEffect(() => {
    targetRmsRef.current = rms;
    const upper = frequencies.slice(6, 14);
    const avgHigh =
      upper.length > 0
        ? upper.reduce((acc, v) => acc + v, 0) / upper.length
        : 0;
    targetHighFreqRef.current = avgHigh;
    statusRef.current = status;
  }, [status, rms, frequencies]);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.z = 3.2;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setSize(220, 220);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    const geometry = new THREE.IcosahedronGeometry(1.0, 18);
    const uniforms = {
      uTime: { value: 0 },
      uAmplitude: { value: 0.04 },
      uHighFreq: { value: 0.0 },
      uPrimaryColor: { value: new THREE.Color("#0066ff") },
      uSecondaryColor: { value: new THREE.Color("#00f0ff") },
    };

    const material = new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      uniforms,
      transparent: true,
      wireframe: true,
      blending: THREE.AdditiveBlending,
    });

    const orbMesh = new THREE.Mesh(geometry, material);
    scene.add(orbMesh);

    const colorCyan = new THREE.Color("#00f0ff");
    const colorBlue = new THREE.Color("#0066ff");
    const colorGreen = new THREE.Color("#00ff88");
    const colorEmerald = new THREE.Color("#00a86b");

    let animId = 0;
    const clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();
      uniforms.uTime.value = elapsed;

      // Suavização exponencial de amplitude e frequências
      uniforms.uAmplitude.value +=
        (targetRmsRef.current - uniforms.uAmplitude.value) * 0.18;
      uniforms.uHighFreq.value +=
        (targetHighFreqRef.current - uniforms.uHighFreq.value) * 0.18;

      // Transição dinâmica de cores: Verde (LISTENING) vs Ciano/Azul (STANDBY/SPEAKING/PROCESSING)
      const isListening = statusRef.current === "LISTENING";
      const targetPrimary = isListening ? colorEmerald : colorBlue;
      const targetSecondary = isListening ? colorGreen : colorCyan;

      uniforms.uPrimaryColor.value.lerp(targetPrimary, 0.08);
      uniforms.uSecondaryColor.value.lerp(targetSecondary, 0.08);

      orbMesh.rotation.y = elapsed * 0.32;
      orbMesh.rotation.x = Math.sin(elapsed * 0.4) * 0.18;

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      cancelAnimationFrame(animId);
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    };
  }, []);

  return (
    <div
      ref={mountRef}
      className="w-[220px] h-[220px] flex items-center justify-center pointer-events-none"
    />
  );
};


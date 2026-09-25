"""CPU algebra checks against expressions extracted from the production shader.

Run with Python 3 (stdlib only). These check phase normalization/first moment,
endpoint safety and the shadow quadrature contract, not shader compilation or
GPU precision. GPU mip validation is available via r_cloudsValidateNoiseMips.
"""
import math
from pathlib import Path
import re
import unittest


SHADER = (Path(__file__).resolve().parents[1] /
          "Assets/Shaders/Cloudscape/CloudscapeCS.azsl").read_text()


def expression(name, source=SHADER):
    return re.search(r"(?:const )?float " + name + r"\s*=\s*([^;]+);", source)[1]


HG = SHADER.split("float CalcHenyeyGreenstein(", 1)[1].split("return  henyeyGreenstein;", 1)[0]
PHASE_CODE = {name: compile(expression(name, HG), "shader:" + name, "eval")
              for name in ("g2", "denom", "henyeyGreenstein")}
INV_4PI = eval(re.search(r"#define INV_4PI (.+)", HG)[1], {"__builtins__": {}})
G_LIMITS = tuple(map(float, re.search(r"g = clamp\(g, ([^,]+), ([^)]+)\)", HG).groups()))


def phase(cosine, g):
    env = {"g": min(max(g, G_LIMITS[0]), G_LIMITS[1]),
           "cosAngle": cosine, "sqrt": math.sqrt, "INV_4PI": INV_4PI}
    for name, code in PHASE_CODE.items():
        env[name] = eval(code, {"__builtins__": {}}, env)
    return env["henyeyGreenstein"]


def integrate(f, a=-1.0, b=1.0, tolerance=1e-8):
    def refine(a, b, fa, fm, fb, whole, eps, depth):
        m = (a + b) * 0.5
        fl, fr = f((a + m) * 0.5), f((m + b) * 0.5)
        left = (m - a) * (fa + 4 * fl + fm) / 6
        right = (b - m) * (fm + 4 * fr + fb) / 6
        delta = left + right - whole
        if depth == 0 or abs(delta) <= 15 * eps:
            return left + right + delta / 15
        return (refine(a, m, fa, fl, fm, left, eps / 2, depth - 1) +
                refine(m, b, fm, fr, fb, right, eps / 2, depth - 1))
    fa, fm, fb = f(a), f((a + b) / 2), f(b)
    return refine(a, b, fa, fm, fb, (b-a)*(fa+4*fm+fb)/6, tolerance, 30)


class CloudTransportTests(unittest.TestCase):
    def test_phase_integrates_to_one(self):
        for g in (-0.99, -0.8, -0.3, 0, 0.3, 0.8, 0.99):
            with self.subTest(g=g):
                self.assertAlmostEqual(2 * math.pi * integrate(lambda u: phase(u, g)), 1, places=6)

    def test_phase_mean_cosine_is_g(self):
        for g in (-0.99, -0.8, 0, 0.3, 0.8, 0.99):
            with self.subTest(g=g):
                self.assertAlmostEqual(2 * math.pi * integrate(lambda u: u * phase(u, g)), g, places=6)

    def test_isotropic_and_forward_convention(self):
        for cosine in (-1, -0.5, 0, 0.5, 1):
            self.assertAlmostEqual(phase(cosine, 0), 1/(4*math.pi), places=12)
        self.assertGreater(phase(1, 0.8), phase(-1, 0.8))

    def test_endpoint_inputs_remain_finite(self):
        for g in (-2, -1, -0.99, 0.99, 1, 2):
            for cosine in (-1, 0, 1):
                self.assertTrue(math.isfinite(phase(cosine, g)))
                self.assertGreater(phase(cosine, g), 0)

    def test_shadow_segments_cover_physical_distance(self):
        body = SHADER.split("float3 GetMultiScatteredLuminance(", 1)[1].split("struct AtmosphereIntersectionInfo", 1)[0]
        count = int(re.search(r"#define NUM_LIGHT_SAMPLES \((\d+)\)", body)[1])
        # A homogeneous medium must integrate to sigma_t * rho * distance.
        for extent in (0.1, 1, 3.5, 10):
            env = {"shadowDistanceKm": extent, "NUM_LIGHT_SAMPLES": count, "float": float}
            step = eval(expression("baseShadowStepKm", body), {"__builtins__": {}}, env)
            start = 0
            optical_depth = 0
            for i in range(count):
                env.update(baseShadowStepKm=step, distanceMultipler=2**i, segmentStartKm=start)
                width = eval(expression("lightStepDistance", body), {"__builtins__": {}}, env)
                env["lightStepDistance"] = width
                sample = eval(expression("sampleDistanceKm", body), {"__builtins__": {}}, env)
                self.assertLess(start, sample)
                self.assertLess(sample, start+width)
                optical_depth += 0.7 * 4 * width
                start += width
            self.assertAlmostEqual(start, extent)
            self.assertAlmostEqual(math.exp(-optical_depth), math.exp(-0.7 * 4 * extent))
        self.assertNotIn("stepSizeKm", body)
        self.assertNotIn("RayMarchingSteps", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)

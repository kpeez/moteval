pub const EPS: f64 = f64::EPSILON;

pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * pow2(-53)
    }

    pub fn spread(&mut self) -> f64 {
        let negative = self.next_u64() & 1 == 1;
        let mantissa = self.uniform();
        let exponent = self.below(86) as i32 - 30;
        let value = mantissa * pow2(exponent);
        if negative { -value } else { value }
    }
}

pub fn pow2(k: i32) -> f64 {
    assert!((-1022..=1023).contains(&k));
    f64::from_bits(((1023 + k) as u64) << 52)
}

pub fn fnv1a64<'a>(values: impl IntoIterator<Item = &'a f64>) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for x in values {
        for byte in x.to_le_bytes() {
            h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

pub fn pattern(name: &str, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    match name {
        "alternating" => {
            let cycle = [1e16, 1.0, -1e16, 1.0];
            (0..n).map(|i| cycle[i % 4]).collect()
        }
        "negzero" => vec![-0.0; n],
        "spread" => (0..n).map(|_| rng.spread()).collect(),
        "uniform" => (0..n).map(|_| rng.uniform()).collect(),
        "big_small" => (0..n)
            .map(|_| {
                let u = rng.next_u64();
                let value = if u & 1 == 1 { 1e16 } else { 1.0 };
                if u & 2 == 2 { -value } else { value }
            })
            .collect(),
        "nan_spread" => (0..n)
            .map(|_| {
                let value = rng.spread();
                if rng.below(5) == 0 { f64::NAN } else { value }
            })
            .collect(),
        other => panic!("unknown pattern {other:?}"),
    }
}

pub fn random_box(rng: &mut SplitMix64) -> [f64; 4] {
    let mut x = rng.below(40) as f64 * 0.5;
    let mut y = rng.below(40) as f64 * 0.5;
    let mut w = rng.below(24) as f64 * 0.5;
    let mut h = rng.below(24) as f64 * 0.5;
    if rng.below(3) == 0 {
        x += rng.uniform();
        w += rng.uniform();
    }
    if rng.below(4) == 0 {
        y += rng.uniform() * 3.0;
        h += rng.uniform();
    }
    if rng.below(10) == 0 {
        w = -w;
    }
    if rng.below(12) == 0 {
        x += 1_048_576.0;
    }
    if rng.below(15) == 0 {
        x = -0.0;
    }
    [x, y, w, h]
}

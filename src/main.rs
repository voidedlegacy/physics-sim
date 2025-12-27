use std::f32::consts::PI;

#[derive(Clone, Copy, Debug, Default)]
struct Vec2 {
    x: f32,
    y: f32,
}

impl Vec2 {
    fn new(x: f32, y: f32) -> Self { Self { x, y } }
    fn dot(self, o: Vec2) -> f32 { self.x * o.x + self.y * o.y }
    fn len2(self) -> f32 { self.dot(self) }
    fn len(self) -> f32 { self.len2().sqrt() }
    fn normalized(self) -> Vec2 {
        let l = self.len();
        if l > 1e-8 { self / l } else { Vec2::new(0.0, 0.0) }
    }
}

use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, rhs: Vec2) -> Vec2 { Vec2::new(self.x + rhs.x, self.y + rhs.y) }
}
impl AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Vec2) { *self = *self + rhs; }
}
impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, rhs: Vec2) -> Vec2 { Vec2::new(self.x - rhs.x, self.y - rhs.y) }
}
impl SubAssign for Vec2 {
    fn sub_assign(&mut self, rhs: Vec2) { *self = *self - rhs; }
}
impl Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, rhs: f32) -> Vec2 { Vec2::new(self.x * rhs, self.y * rhs) }
}
impl Div<f32> for Vec2 {
    type Output = Vec2;
    fn div(self, rhs: f32) -> Vec2 { Vec2::new(self.x / rhs, self.y / rhs) }
}

#[derive(Clone, Copy, Debug)]
struct Body {
    pos: Vec2,
    vel: Vec2,
    radius: f32,
    inv_mass: f32,     // 0 => infinite mass (static)
    restitution: f32,  // bounciness: 0..1
}

impl Body {
    fn circle(pos: Vec2, vel: Vec2, radius: f32, mass: f32, restitution: f32) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self { pos, vel, radius, inv_mass, restitution }
    }

    fn mass(&self) -> f32 {
        if self.inv_mass == 0.0 { f32::INFINITY } else { 1.0 / self.inv_mass }
    }

    // Not used, but handy if you want angular physics later.
    fn area(&self) -> f32 {
        PI * self.radius * self.radius
    }
}

struct World {
    bodies: Vec<Body>,
    gravity: Vec2,
    bounds_min: Vec2,
    bounds_max: Vec2,
    // small position correction factor to reduce sinking/jitter:
    baumgarte: f32,
}

impl World {
    fn new(bounds_min: Vec2, bounds_max: Vec2) -> Self {
        Self {
            bodies: Vec::new(),
            gravity: Vec2::new(0.0, -9.81),
            bounds_min,
            bounds_max,
            baumgarte: 0.2,
        }
    }

    fn add(&mut self, body: Body) {
        self.bodies.push(body);
    }

    fn step(&mut self, dt: f32, substeps: usize) {
        let h = dt / substeps as f32;
        for _ in 0..substeps {
            self.integrate(h);
            self.solve_collisions();
            self.solve_bounds();
        }
    }

    fn integrate(&mut self, dt: f32) {
        for b in &mut self.bodies {
            if b.inv_mass == 0.0 { continue; } // static body
            // semi-implicit Euler: v += a*dt, x += v*dt
            b.vel += self.gravity * dt;
            b.pos += b.vel * dt;
        }
    }

    fn solve_bounds(&mut self) {
        for b in &mut self.bodies {
            if b.inv_mass == 0.0 { continue; }

            // left wall
            if b.pos.x - b.radius < self.bounds_min.x {
                b.pos.x = self.bounds_min.x + b.radius;
                b.vel.x = -b.vel.x * b.restitution;
            }
            // right wall
            if b.pos.x + b.radius > self.bounds_max.x {
                b.pos.x = self.bounds_max.x - b.radius;
                b.vel.x = -b.vel.x * b.restitution;
            }
            // floor
            if b.pos.y - b.radius < self.bounds_min.y {
                b.pos.y = self.bounds_min.y + b.radius;
                b.vel.y = -b.vel.y * b.restitution;
            }
            // ceiling
            if b.pos.y + b.radius > self.bounds_max.y {
                b.pos.y = self.bounds_max.y - b.radius;
                b.vel.y = -b.vel.y * b.restitution;
            }
        }
    }

    fn solve_collisions(&mut self) {
        let n = self.bodies.len();
        for i in 0..n {
            for j in (i + 1)..n {
                // (Borrow checker friendly) pull copies, then write back.
                let (a, b) = {
                    let a = self.bodies[i];
                    let b = self.bodies[j];
                    (a, b)
                };

                // Skip if both static
                if a.inv_mass == 0.0 && b.inv_mass == 0.0 {
                    continue;
                }

                let delta = b.pos - a.pos;
                let dist2 = delta.len2();
                let r = a.radius + b.radius;

                if dist2 >= r * r {
                    continue; // no overlap
                }

                let dist = dist2.sqrt().max(1e-6);
                let normal = delta / dist; // from A to B

                // Positional correction (separate them)
                let penetration = r - dist;
                let inv_mass_sum = a.inv_mass + b.inv_mass;
                if inv_mass_sum > 0.0 {
                    let correction = normal * (penetration * self.baumgarte / inv_mass_sum);
                    let mut a2 = a;
                    let mut b2 = b;
                    if a2.inv_mass > 0.0 { a2.pos -= correction * a2.inv_mass; }
                    if b2.inv_mass > 0.0 { b2.pos += correction * b2.inv_mass; }

                    // Relative velocity
                    let rv = b2.vel - a2.vel;
                    let vel_along_normal = rv.dot(normal);

                    // If they're separating, don't bounce
                    if vel_along_normal < 0.0 {
                        let e = a2.restitution.min(b2.restitution); // combine restitutions
                        let j_impulse = -(1.0 + e) * vel_along_normal / inv_mass_sum;
                        let impulse = normal * j_impulse;

                        if a2.inv_mass > 0.0 { a2.vel -= impulse * a2.inv_mass; }
                        if b2.inv_mass > 0.0 { b2.vel += impulse * b2.inv_mass; }
                    }

                    // Write back
                    self.bodies[i] = a2;
                    self.bodies[j] = b2;
                }
            }
        }
    }
}

fn main() {
    // A "box" world from (0,0) to (10,6)
    let mut world = World::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 6.0));

    // Add a few circles
    world.add(Body::circle(
        Vec2::new(2.0, 5.0),
        Vec2::new(2.0, 0.0),
        0.30,
        1.0,
        0.65,
    ));
    world.add(Body::circle(
        Vec2::new(4.0, 5.2),
        Vec2::new(-1.0, 0.0),
        0.35,
        1.2,
        0.65,
    ));
    world.add(Body::circle(
        Vec2::new(6.0, 5.5),
        Vec2::new(0.0, 0.0),
        0.40,
        2.0,
        0.55,
    ));

    // Simulate ~5 seconds at 60 Hz
    let dt = 1.0 / 60.0;
    let steps = (5.0 / dt) as usize;

    for frame in 0..steps {
        world.step(dt, 4); // 4 substeps for stability

        // Print every 10 frames
        if frame % 10 == 0 {
            println!("frame {frame}");
            for (i, b) in world.bodies.iter().enumerate() {
                println!(
                    "  body {i}: pos=({:6.3}, {:6.3}) vel=({:6.3}, {:6.3})",
                    b.pos.x, b.pos.y, b.vel.x, b.vel.y
                );
            }
        }
    }
}
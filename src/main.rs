use pixels::{Pixels, SurfaceTexture};
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalPosition, PhysicalSize},
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const SIM_W: u32 = 640;
const SIM_H: u32 = 360;

// Physics tuning
const G: f32 = 2200.0;
const SOFTEN: f32 = 40.0;
const RESTITUTION: f32 = 0.85;
const DRAG: f32 = 0.999;
const MAX_SPEED: f32 = 2000.0;

#[derive(Clone, Copy, Debug)]
struct Body {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    r: f32,
    m: f32,
}
impl Body {
    fn new(x: f32, y: f32, vx: f32, vy: f32, r: f32) -> Self {
        let m = (r * r).max(1.0);
        Self { x, y, vx, vy, r, m }
    }
}

#[derive(Default)]
struct World {
    bodies: Vec<Body>,
    paused: bool,
}
impl World {
    fn seed(&mut self) {
        self.bodies.clear();
        self.bodies.push(Body::new(SIM_W as f32 * 0.35, SIM_H as f32 * 0.5, 0.0, -220.0, 9.0));
        self.bodies.push(Body::new(SIM_W as f32 * 0.65, SIM_H as f32 * 0.5, 0.0,  220.0, 9.0));
        self.bodies.push(Body::new(SIM_W as f32 * 0.50, SIM_H as f32 * 0.5, 0.0,    0.0, 14.0));
    }
    fn clear(&mut self) {
        self.bodies.clear();
    }
    fn add_body(&mut self, b: Body) {
        self.bodies.push(b);
    }

    fn step(&mut self, dt: f32) {
        if self.paused || self.bodies.is_empty() {
            return;
        }
        let n = self.bodies.len();
        let mut ax = vec![0.0f32; n];
        let mut ay = vec![0.0f32; n];

        // Pairwise gravity
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.bodies[j].x - self.bodies[i].x;
                let dy = self.bodies[j].y - self.bodies[i].y;
                let dist2 = dx * dx + dy * dy + SOFTEN * SOFTEN;
                let inv_dist = 1.0 / dist2.sqrt();
                let inv_dist3 = inv_dist * inv_dist * inv_dist;

                let f = G * self.bodies[i].m * self.bodies[j].m * inv_dist3;

                let fx = f * dx;
                let fy = f * dy;

                ax[i] += fx / self.bodies[i].m;
                ay[i] += fy / self.bodies[i].m;
                ax[j] -= fx / self.bodies[j].m;
                ay[j] -= fy / self.bodies[j].m;
            }
        }

        // Integrate + walls
        for i in 0..n {
            let b = &mut self.bodies[i];
            b.vx = (b.vx + ax[i] * dt) * DRAG;
            b.vy = (b.vy + ay[i] * dt) * DRAG;

            let sp2 = b.vx * b.vx + b.vy * b.vy;
            if sp2 > MAX_SPEED * MAX_SPEED {
                let s = MAX_SPEED / sp2.sqrt();
                b.vx *= s;
                b.vy *= s;
            }

            b.x += b.vx * dt;
            b.y += b.vy * dt;

            if b.x - b.r < 0.0 {
                b.x = b.r;
                b.vx = -b.vx * RESTITUTION;
            }
            if b.x + b.r > SIM_W as f32 {
                b.x = SIM_W as f32 - b.r;
                b.vx = -b.vx * RESTITUTION;
            }
            if b.y - b.r < 0.0 {
                b.y = b.r;
                b.vy = -b.vy * RESTITUTION;
            }
            if b.y + b.r > SIM_H as f32 {
                b.y = SIM_H as f32 - b.r;
                b.vy = -b.vy * RESTITUTION;
            }
        }

        // Circle-circle collisions (naive)
        for i in 0..n {
            for j in (i + 1)..n {
                let (xi, yi, ri, mi) = {
                    let b = self.bodies[i];
                    (b.x, b.y, b.r, b.m)
                };
                let (xj, yj, rj, mj) = {
                    let b = self.bodies[j];
                    (b.x, b.y, b.r, b.m)
                };

                let dx = xj - xi;
                let dy = yj - yi;
                let dist = (dx * dx + dy * dy).sqrt();
                let min_dist = ri + rj;

                if dist > 0.0 && dist < min_dist {
                    let nx = dx / dist;
                    let ny = dy / dist;

                    let penetration = min_dist - dist;
                    let total_m = mi + mj;
                    let move_i = penetration * (mj / total_m);
                    let move_j = penetration * (mi / total_m);

                    self.bodies[i].x -= nx * move_i;
                    self.bodies[i].y -= ny * move_i;
                    self.bodies[j].x += nx * move_j;
                    self.bodies[j].y += ny * move_j;

                    let rvx = self.bodies[j].vx - self.bodies[i].vx;
                    let rvy = self.bodies[j].vy - self.bodies[i].vy;
                    let vel_along_normal = rvx * nx + rvy * ny;

                    if vel_along_normal < 0.0 {
                        let e = RESTITUTION;
                        let inv_mi = 1.0 / mi;
                        let inv_mj = 1.0 / mj;
                        let j_impulse = -(1.0 + e) * vel_along_normal / (inv_mi + inv_mj);

                        let ix = j_impulse * nx;
                        let iy = j_impulse * ny;

                        self.bodies[i].vx -= ix * inv_mi;
                        self.bodies[i].vy -= iy * inv_mi;
                        self.bodies[j].vx += ix * inv_mj;
                        self.bodies[j].vy += iy * inv_mj;
                    }
                }
            }
        }
    }
}

// ---------- Drawing ----------
fn clear(frame: &mut [u8]) {
    for px in frame.chunks_exact_mut(4) {
        px[0] = 8;
        px[1] = 10;
        px[2] = 16;
        px[3] = 255;
    }
}
fn put_pixel(frame: &mut [u8], w: u32, x: i32, y: i32, rgba: [u8; 4]) {
    if x < 0 || y < 0 {
        return;
    }
    let (x, y) = (x as u32, y as u32);
    if x >= w {
        return;
    }
    let idx = ((y * w + x) * 4) as usize;
    if idx + 3 >= frame.len() {
        return;
    }
    frame[idx..idx + 4].copy_from_slice(&rgba);
}
fn draw_filled_circle(frame: &mut [u8], w: u32, h: u32, cx: f32, cy: f32, r: f32, rgba: [u8; 4]) {
    let r2 = r * r;
    let min_x = (cx - r).floor() as i32;
    let max_x = (cx + r).ceil() as i32;
    let min_y = (cy - r).floor() as i32;
    let max_y = (cy + r).ceil() as i32;

    for y in min_y..=max_y {
        if y < 0 || y >= h as i32 {
            continue;
        }
        for x in min_x..=max_x {
            if x < 0 || x >= w as i32 {
                continue;
            }
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            if dx * dx + dy * dy <= r2 {
                put_pixel(frame, w, x, y, rgba);
            }
        }
    }
}

type Error = Box<dyn std::error::Error>;
type Result<T> = std::result::Result<T, Error>;

#[derive(Default)]
struct App {
    window: Option<&'static Window>,
    pixels: Option<Pixels<'static>>,

    world: World,
    cursor_px: (usize, usize),
    drag_start: Option<(usize, usize)>,

    last: Option<Instant>,
    acc: f32,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create window inside the running event loop (winit 0.30 style).
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Rust Physics Sim (winit 0.30 + pixels)")
                    .with_inner_size(LogicalSize::new((SIM_W * 2) as f64, (SIM_H * 2) as f64)),
            )
            .unwrap();

        // pixels wants the window handle to outlive it; simplest hack is to leak the window.
        let window_ref: &'static Window = Box::leak(Box::new(window));

        let size = window_ref.inner_size();
        let surface = SurfaceTexture::new(size.width, size.height, window_ref);
        let pixels = Pixels::new(SIM_W, SIM_H, surface).unwrap();

        self.window = Some(window_ref);
        self.pixels = Some(pixels);

        self.cursor_px = (SIM_W as usize / 2, SIM_H as usize / 2);
        self.drag_start = None;

        self.world.seed();
        self.last = Some(Instant::now());
        self.acc = 0.0;
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(window) = self.window else { return; };
        let Some(pixels) = self.pixels.as_mut() else { return; };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(PhysicalSize { width, height }) => {
                let _ = pixels.resize_surface(width, height);
            }

            WindowEvent::ScaleFactorChanged { .. } => {
                let size = window.inner_size();
                let _ = pixels.resize_surface(size.width, size.height);
            }

            WindowEvent::CursorMoved { position, .. } => {
                let PhysicalPosition { x, y } = position;
                let cursor_position: (f32, f32) = (x as f32, y as f32);

                self.cursor_px = pixels
                    .window_pos_to_pixel(cursor_position)
                    .unwrap_or_else(|pos| pixels.clamp_pixel_pos(pos));
            }

            WindowEvent::MouseInput { state, button, .. } => match (button, state) {
                (MouseButton::Left, ElementState::Pressed) => {
                    self.drag_start = Some(self.cursor_px);
                }
                (MouseButton::Left, ElementState::Released) => {
                    if let Some(start) = self.drag_start.take() {
                        let end = self.cursor_px;
                        let vx = (start.0 as f32 - end.0 as f32) * 3.0;
                        let vy = (start.1 as f32 - end.1 as f32) * 3.0;
                        self.world.add_body(Body::new(start.0 as f32, start.1 as f32, vx, vy, 8.0));
                    }
                }
                (MouseButton::Right, ElementState::Pressed) => {
                    self.world
                        .add_body(Body::new(self.cursor_px.0 as f32, self.cursor_px.1 as f32, 0.0, 0.0, 14.0));
                }
                _ => {}
            },

            WindowEvent::KeyboardInput { event: KeyEvent { physical_key, state, repeat, .. }, .. } => {
                if state == ElementState::Pressed && !repeat {
                    match physical_key {
                        PhysicalKey::Code(KeyCode::Space) => self.world.paused = !self.world.paused,
                        PhysicalKey::Code(KeyCode::KeyR) => self.world.seed(),
                        PhysicalKey::Code(KeyCode::KeyC) => self.world.clear(),
                        PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                        _ => {}
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let frame = pixels.frame_mut();
                clear(frame);

                for (i, b) in self.world.bodies.iter().enumerate() {
                    let c = match i % 5 {
                        0 => [220, 80, 80, 255],
                        1 => [80, 220, 140, 255],
                        2 => [80, 140, 220, 255],
                        3 => [220, 200, 80, 255],
                        _ => [200, 120, 220, 255],
                    };
                    draw_filled_circle(frame, SIM_W, SIM_H, b.x, b.y, b.r, c);
                }

                if let Some(start) = self.drag_start {
                    let (sx, sy) = (start.0 as i32, start.1 as i32);
                    let (ex, ey) = (self.cursor_px.0 as i32, self.cursor_px.1 as i32);
                    let steps = 40;
                    for t in 0..=steps {
                        let a = t as f32 / steps as f32;
                        let x = (sx as f32 + (ex - sx) as f32 * a) as i32;
                        let y = (sy as f32 + (ey - sy) as f32 * a) as i32;
                        put_pixel(frame, SIM_W, x, y, [255, 255, 255, 220]);
                    }
                }

                // draw to window
                let _ = pixels.render();
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Fixed timestep update + continuous animation
        let Some(window) = self.window else { return; };

        let now = Instant::now();
        let last = self.last.get_or_insert(now);
        let mut frame_dt = (now - *last).as_secs_f32();
        *last = now;

        if frame_dt > 0.1 {
            frame_dt = 0.1;
        }

        let dt = 1.0 / 120.0;
        self.acc += frame_dt;
        while self.acc >= dt {
            self.world.step(dt);
            self.acc -= dt;
        }

        window.request_redraw();
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app)?;

    Ok(())
}

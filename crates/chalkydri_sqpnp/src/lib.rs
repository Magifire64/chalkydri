use nalgebra::{
    Isometry3, Point3, Rotation3, SMatrix, SVector, Translation3, UnitQuaternion, Vector3,
};
use std::ops::AddAssign;

// --- Type Definitions ---
pub type Mat3 = SMatrix<f64, 3, 3>;
pub type Mat9 = SMatrix<f64, 9, 9>;
pub type Vec9 = SVector<f64, 9>;
pub type Mat15 = SMatrix<f64, 15, 15>;
pub type Vec15 = SVector<f64, 15>;
pub type Mat6x9 = SMatrix<f64, 6, 9>;
pub type Vec6 = SVector<f64, 6>;
pub type Vec3 = SVector<f64, 3>;
pub type Mat9x3 = SMatrix<f64, 9, 3>;
pub type Iso3 = Isometry3<f64>;
pub type Pnt3 = Point3<f64>;
pub type Rot3 = Rotation3<f64>;

// --- Helper: Define Camera Offset ---

/// Creates the transform defining where the Camera is located on the Robot.
/// Returns the Isometry T_{Robot <- Camera}
pub fn make_camera_pose(mounting_pitch_deg: f64, robot_center_to_cam: Vec3) -> Iso3 {
    // 1. Define the rotation from Robot Frame (NWU) to Camera Optical Frame (Z-Fwd)
    let nwu_to_optical = Mat3::new(0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0);
    let base_rot = Rotation3::from_matrix(&nwu_to_optical);

    // 2. Apply Pitch (Tilt Up/Down)
    // Positive mounting pitch (looking up) is a Negative rotation around Y (Right Hand Rule)
    let pitch_rad = -mounting_pitch_deg.to_radians();
    let pitch_rot = Rotation3::from_axis_angle(&Vector3::y_axis(), pitch_rad);

    let final_rot = pitch_rot * base_rot;

    // FIX: Convert Rotation3 to UnitQuaternion for Isometry3
    let quat = UnitQuaternion::from_rotation_matrix(&final_rot);

    Iso3::from_parts(Translation3::from(robot_center_to_cam), quat)
}

// --- SQPnP Solver Implementation ---

#[inline(always)]
fn nearest_so3(r_vec: &Vec9) -> Vec9 {
    let m = Mat3::from_column_slice(r_vec.as_slice());
    let svd = m.svd(true, true);
    let u = svd.u.unwrap_or_default();
    let vt = svd.v_t.unwrap_or_default();

    let mut rot = u * vt;
    if rot.determinant() < 0.0 {
        let mut u_prime = u;
        u_prime.column_mut(2).scale_mut(-1.0);
        rot = u_prime * vt;
    }
    Vec9::from_column_slice(rot.as_slice())
}

#[inline(always)]
fn constraints_and_jacobian(r_vec: &Vec9) -> (Vec6, Mat6x9) {
    let c1 = r_vec.fixed_view::<3, 1>(0, 0);
    let c2 = r_vec.fixed_view::<3, 1>(3, 0);
    let c3 = r_vec.fixed_view::<3, 1>(6, 0);

    let h = Vec6::new(
        c1.norm_squared() - 1.0,
        c2.norm_squared() - 1.0,
        c3.norm_squared() - 1.0,
        c1.dot(&c2),
        c1.dot(&c3),
        c2.dot(&c3),
    );

    let mut jac = Mat6x9::zeros();

    jac.fixed_view_mut::<1, 3>(0, 0)
        .copy_from(&(2.0 * c1.transpose()));
    jac.fixed_view_mut::<1, 3>(1, 3)
        .copy_from(&(2.0 * c2.transpose()));
    jac.fixed_view_mut::<1, 3>(2, 6)
        .copy_from(&(2.0 * c3.transpose()));
    jac.fixed_view_mut::<1, 3>(3, 0).copy_from(&c2.transpose());
    jac.fixed_view_mut::<1, 3>(3, 3).copy_from(&c1.transpose());
    jac.fixed_view_mut::<1, 3>(4, 0).copy_from(&c3.transpose());
    jac.fixed_view_mut::<1, 3>(4, 6).copy_from(&c1.transpose());
    jac.fixed_view_mut::<1, 3>(5, 3).copy_from(&c3.transpose());
    jac.fixed_view_mut::<1, 3>(5, 6).copy_from(&c2.transpose());

    (h, jac)
}

#[inline(always)]
fn solve_newton(r: &Vec9, omega: &Mat9, h: &Vec6, jac: &Mat6x9) -> Option<Vec9> {
    let mut lhs = Mat15::zeros();
    lhs.fixed_view_mut::<9, 9>(0, 0).copy_from(omega);
    lhs.fixed_view_mut::<9, 6>(0, 9)
        .copy_from(&jac.transpose());
    lhs.fixed_view_mut::<6, 9>(9, 0).copy_from(jac);

    let mut rhs = Vec15::zeros();
    let omega_r = omega * r;

    rhs.fixed_view_mut::<9, 1>(0, 0).copy_from(&(-omega_r));
    rhs.fixed_view_mut::<6, 1>(9, 0).copy_from(&(-h));

    match lhs.lu().solve(&rhs) {
        Some(sol) => Some(sol.fixed_view::<9, 1>(0, 0).into_owned()),
        None => None,
    }
}

struct LinearSys {
    omega: Mat9,
    q_tt_inv: Mat3,
    q_rt: Mat9x3,
}

#[inline(always)]
fn build_linear_system(points_3d: &[Vec3], points_2d: &[Vec3]) -> LinearSys {
    let n = points_3d.len();
    assert_eq!(n, points_2d.len());

    let mut q_rr = Mat9::zeros();
    let mut q_rt = Mat9x3::zeros();
    let mut q_tt = Mat3::zeros();

    for (p_3d, p_img) in points_3d.iter().zip(points_2d.iter()) {
        let sq_norm = p_img.norm_squared();
        let inv_norm = 1.0 / sq_norm;
        let v_vt = p_img * p_img.transpose();
        let p = Mat3::identity() - v_vt.scale(inv_norm);

        q_tt += p;

        let px = p.scale(p_3d.x);
        let py = p.scale(p_3d.y);
        let pz = p.scale(p_3d.z);

        q_rt.fixed_view_mut::<3, 3>(0, 0).add_assign(&px);
        q_rt.fixed_view_mut::<3, 3>(3, 0).add_assign(&py);
        q_rt.fixed_view_mut::<3, 3>(6, 0).add_assign(&pz);

        q_rr.fixed_view_mut::<3, 3>(0, 0)
            .add_assign(&px.scale(p_3d.x));
        q_rr.fixed_view_mut::<3, 3>(3, 3)
            .add_assign(&py.scale(p_3d.y));
        q_rr.fixed_view_mut::<3, 3>(6, 6)
            .add_assign(&pz.scale(p_3d.z));

        let pxy = px.scale(p_3d.y);
        q_rr.fixed_view_mut::<3, 3>(0, 3).add_assign(&pxy);
        q_rr.fixed_view_mut::<3, 3>(3, 0).add_assign(&pxy);

        let pxz = px.scale(p_3d.z);
        q_rr.fixed_view_mut::<3, 3>(0, 6).add_assign(&pxz);
        q_rr.fixed_view_mut::<3, 3>(6, 0).add_assign(&pxz);

        let pyz = py.scale(p_3d.z);
        q_rr.fixed_view_mut::<3, 3>(3, 6).add_assign(&pyz);
        q_rr.fixed_view_mut::<3, 3>(6, 3).add_assign(&pyz);
    }

    let q_tt_inv = q_tt.try_inverse().unwrap_or_default();
    let temp = q_rt * q_tt_inv;
    let omega = q_rr - temp * q_rt.transpose();

    LinearSys {
        omega,
        q_tt_inv,
        q_rt,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SqPnP {
    max_iter: usize,
    tol_sq: f64,
    corner_distance: f64,
}

impl Default for SqPnP {
    fn default() -> Self {
        Self {
            max_iter: 15,
            tol_sq: 1e-16,
            corner_distance: 0.1651f64 / 2.0,
        }
    }
}

impl SqPnP {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub const fn tolerance(mut self, tol: f64) -> Self {
        self.tol_sq = tol * tol;
        self
    }

    pub fn with_tag_side_size(mut self, size: f64) -> Self {
        self.corner_distance = size / 2.0;
        self
    }

    /// Solves for the standard Computer Vision pose (World -> Camera).
    /// Returns Isometry T_{Cam <- World}
    fn solve(
        &self,
        points_isometry: &[Isometry3<f64>],
        points_2d: &[Vec3],
        gyro: f64,
        sign_change_error: f64,
        buffer: &mut Vec<Pnt3>,
    ) -> Option<Iso3> {
        self.corner_points_from_center(points_isometry, buffer);

        let mut points_3d: Vec<Vec3> = Vec::with_capacity(buffer.len());
        for point in buffer {
            points_3d.push(Vec3::new(point.x, point.y, point.z));
        }

        if points_3d.len() < 3 || points_3d.len() != points_2d.len() {
            return None;
        }

        let sys = build_linear_system(&points_3d, points_2d);

        let r_mat = self.solve_rotation(&sys.omega, gyro, sign_change_error);

        let r_vec = Vec9::from_column_slice(r_mat.as_slice());
        let t_vec = -sys.q_tt_inv * sys.q_rt.transpose() * r_vec;

        // FIX: Create proper Rotation3 object first
        let rot = Rot3::from_matrix(&r_mat);
        // FIX: Convert Rotation3 to UnitQuaternion for Isometry3
        let quat = UnitQuaternion::from_rotation_matrix(&rot);

        Some(Iso3::from_parts(Translation3::from(t_vec), quat))
    }

    /// Solves for the Robot's Pose in the World.
    /// Returns Isometry T_{World <- Robot}
    pub fn solve_robot_pose(
        &self,
        points_isometry: &[Isometry3<f64>],
        points_2d: &[Vec3],
        gyro: f64,
        sign_change_error: f64,
        buffer: &mut Vec<Pnt3>,
        cam_on_robot: &Iso3,
    ) -> Option<Iso3> {
        // 1. Solve PnP to get T_{Cam <- World}
        let iso_world_to_cam = self.solve(points_isometry, points_2d, gyro, sign_change_error, buffer)?;

        // 2. Invert to get T_{World <- Cam}
        let iso_cam_to_world = iso_world_to_cam.inverse();

        // 3. T_{World <- Robot} = T_{World <- Cam} * (T_{Robot <- Cam})^-1
        let iso_cam_on_robot_inv = cam_on_robot.inverse();
        let iso_robot_in_world = iso_cam_to_world * iso_cam_on_robot_inv;

        Some(iso_robot_in_world)
    }

    fn corner_points_from_center(&self, isometry: &[Iso3], buffer: &mut Vec<Pnt3>) {
        buffer.clear();
        let s = self.corner_distance;
        isometry.iter().for_each(|iso: &Iso3| {
            // Coordinate System: Y-Down Positive, X-Right Positive.
            // Order: Bottom Left -> Clockwise
            let corners = [
                Pnt3::new(-s, s, 0.0),  // Bottom-Left
                Pnt3::new(-s, -s, 0.0), // Top-Left
                Pnt3::new(s, -s, 0.0),  // Top-Right
                Pnt3::new(s, s, 0.0),   // Bottom-Right
            ];

            for c in corners {
                buffer.push(iso * c);
            }
        });
    }

    fn solve_rotation(&self, omega: &Mat9, gyro: f64, sign_change_error: f64) -> Mat3 {
        let eigen = omega.symmetric_eigen();

        let mut best_r = Vec9::zeros();
        let mut min_energy = f64::MAX;

        for i in 0..3 {
            let e = eigen.eigenvectors.column(i);
            for sign in [-1.0, 1.0] {
                let guess = e.scale(sign);
                let r_start = nearest_so3(&guess);
                let (refined_r, mut energy) = self.optimization(r_start, omega);

                let test_r_mat = Mat3::from_column_slice(refined_r.as_slice());

                // Gyro Ambiguity Check
                let cam_fwd_x_world = test_r_mat[(2, 0)];
                let cam_fwd_y_world = test_r_mat[(2, 1)];

                let dot = (cam_fwd_x_world * gyro.cos()) + (cam_fwd_y_world * gyro.sin());

                if dot < 0.0 {
                    energy += sign_change_error;
                }

                if energy < min_energy {
                    min_energy = energy;
                    best_r = refined_r;
                }
            }
        }
        Mat3::from_column_slice(best_r.as_slice())
    }

    fn optimization(&self, start_r: Vec9, omega: &Mat9) -> (Vec9, f64) {
        let mut r = start_r;
        for _ in 0..self.max_iter {
            let (h, jac) = constraints_and_jacobian(&r);
            match solve_newton(&r, omega, &h, &jac) {
                Some(delta_r) => {
                    r += delta_r;
                    if delta_r.norm_squared() < self.tol_sq {
                        break;
                    }
                }
                None => break,
            }
        }
        let energy = r.dot(&(omega * r));
        (r, energy)
    }
}
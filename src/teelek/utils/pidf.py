import math
import time

class PIDF:
    def __init__(self, min_val, max_val, Kp=0.0, Ki=0.0, i_min=0.0, i_max=0.0, Kd=0.0, Kf=0.0, error_tolerance=0.0):
        # Gains
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kf = Kf
        self.error_tolerance = error_tolerance

        # Output & Integral clamp
        self.out_min = min_val
        self.out_max = max_val
        self.i_min = i_min
        self.i_max = i_max

        # States
        self.setpoint = 0.0
        self.last_error = 0.0
        self.integral = 0.0

        # Derivative filter
        self.d_filt = 0.0        # Filtered derivative
        self.d_fc_hz = 0.0       # Cutoff frequency (Hz), 0 = disabled
        self.d_init = True       # Flag for first-time derivative init

        # Timing
        self.last_time = None

    def set_pidf(self, Kp, Ki, Kd, Kf, error_tolerance):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kf = Kf
        self.error_tolerance = error_tolerance

    def set_output_limits(self, min_val, max_val):
        self.out_min = min_val
        self.out_max = max_val

    def set_i_clamp(self, i_min, i_max):
        self.i_min = i_min
        self.i_max = i_max

    def set_d_filter_cutoff_hz(self, fc_hz):
        self.d_fc_hz = max(0.0, fc_hz)
        self.d_init = True

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.d_filt = 0.0
        self.d_init = True
        self.last_time = None

    def _step_dt(self):
        now = time.time()
        if self.last_time is None:
            self.last_time = now
            return 0.0
        
        dt = now - self.last_time
        self.last_time = now
        
        # ป้องกัน dt ที่น้อยเกินไปจนทำให้เกิด Division by zero หรือค่าพุ่ง
        dt_min = 1e-4 
        return max(dt, dt_min)

    def compute(self, setpoint, measure):
        self.setpoint = setpoint
        error = setpoint - measure
        return self.compute_with_error(error)

    def compute_with_error(self, error):
        dt = self._step_dt()

        # กรณีไม่มี Feedforward และอยู่ในช่วง Error Tolerance (Deadband)
        if self.Kf == 0.0:
            if abs(error) <= self.error_tolerance:
                self.integral = 0.0
                self.last_error = error
                if self.d_init:
                    self.d_filt = 0.0
                else:
                    self.d_filt *= 0.9 # ค่อยๆ ลดค่า D ลง
                return 0.0

        # I-term calculation with clamping
        self.integral += error * dt
        if not (self.i_max == -1 and self.i_min == -1):
            self.integral = max(self.i_min, min(self.i_max, self.integral))

        # D-term (raw calculation)
        d_raw = (error - self.last_error) / dt if dt > 0.0 else 0.0

        # Low-pass derivative filter: alpha = exp(-2*pi*fc*dt)
        d_use = d_raw
        if self.Kd != 0.0 and self.d_fc_hz > 0.0:
            alpha = math.exp(-2.0 * math.pi * self.d_fc_hz * dt)
            alpha = max(0.0, min(1.0, alpha))

            if self.d_init:
                self.d_filt = d_raw
                self.d_init = False
            else:
                self.d_filt = (alpha * self.d_filt) + ((1.0 - alpha) * d_raw)
            d_use = self.d_filt

        # PIDF formula
        out = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * d_use) + (self.Kf * self.setpoint)

        self.last_error = error
        
        # Output Clamping
        return max(self.out_min, min(self.out_max, out))
import math
import time
import queue
import threading
import cv2
import numpy as np
import mediapipe as mp

# ----------------------------- Settings -----------------------------
# Detection / form
UP_ANGLE = 160        # elbow angle considered "arms extended"
DOWN_ANGLE = 90       # elbow angle needed for a full-depth rep
BODY_MIN_ANGLE = 160  # shoulder-hip-ankle angle; below this = hips sagging/piking
SMOOTHING = 0.4       # 0..1, higher = more responsive, lower = smoother
MIN_VISIBILITY = 0.5
COUNT_ONLY_GOOD_REPS = True   # True = only good-form reps increase the count

# Workout plan (sets, goal, rest)
REPS_PER_SET = 10     # target reps in each set
TOTAL_SETS = 3        # number of sets (goal = REPS_PER_SET * TOTAL_SETS)
REST_SECONDS = 30     # rest between sets

# Tempo (seconds per rep, down + up)
MIN_REP_TIME = 1.2    # faster than this -> "Too fast"
MAX_REP_TIME = 6.0    # slower than this -> "Too slow"

# Calories (rough estimate: ~0.32 kcal per push-up for a 70 kg person)
BODY_WEIGHT_KG = 70
KCAL_PER_PUSHUP_70KG = 0.32

# Voice
VOICE_ENABLED = True
WARNING_COOLDOWN = 4.0   # seconds between spoken form warnings

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
L = mp_pose.PoseLandmark


# ----------------------------- Voice feedback -----------------------------
speech_q = queue.Queue()


def _voice_worker():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
    except Exception as e:
        print("Voice disabled (pyttsx3 problem):", repr(e))
        return
    while True:
        text = speech_q.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass


def speak(text):
    """Non-blocking speech; drops messages if too many are waiting."""
    if VOICE_ENABLED and speech_q.qsize() < 3:
        speech_q.put(text)


if VOICE_ENABLED:
    threading.Thread(target=_voice_worker, daemon=True).start()


# ----------------------------- Helpers -----------------------------
def angle(a, b, c):
    """Angle at point b (degrees) formed by points a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cosang, -1.0, 1.0)))


def pick_side(lm):
    """Choose the body side (left/right) the camera sees best."""
    def vis(side):
        ids = [getattr(L, f"{side}_SHOULDER"), getattr(L, f"{side}_ELBOW"),
               getattr(L, f"{side}_WRIST"), getattr(L, f"{side}_HIP"),
               getattr(L, f"{side}_ANKLE")]
        return min(lm[i].visibility for i in ids), ids
    lv, lids = vis("LEFT")
    rv, rids = vis("RIGHT")
    return (lids, lv) if lv >= rv else (rids, rv)


def put(frame, text, pos, color=(0, 255, 0), scale=0.8, thick=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def reset_all():
    global total_reps, bad_reps, set_reps, current_set, phase, stage
    global rest_start, last_rep_time, workout_start
    total_reps = bad_reps = set_reps = 0
    current_set = 1
    phase = "workout"
    stage = "up"
    rest_start = 0.0
    last_rep_time = 0.0
    workout_start = time.time()


SPOKEN_WARNINGS = {"Hips too high!", "Hips sagging!", "Go lower next time",
                   "Too fast, slow down", "Too slow, keep moving",
                   "Bad form - rep not counted"}

# ----------------------------- State -----------------------------
cap = cv2.VideoCapture(0)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

total_reps = bad_reps = set_reps = 0
current_set = 1
phase = "workout"        # "workout", "rest", "done"
stage = "up"             # "up" or "down"
rest_start = 0.0
workout_start = time.time()
rep_start = 0.0
last_rep_time = 0.0
last_warn = 0.0

smooth_elbow = 180.0
smooth_body = 180.0
min_elbow_this_rep = 180.0
rep_form_ok = True
feedback = "Get into push-up position (side view)"
feedback_ok = True

speak("Get ready. Set one starting.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    now = time.time()

    # ---- Rest handling ----
    if phase == "rest":
        remaining = REST_SECONDS - (now - rest_start)
        if remaining <= 0:
            phase = "workout"
            stage = "up"
            feedback, feedback_ok = "Go!", True
            speak(f"Rest over. Start set {current_set}.")

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        (sh_i, el_i, wr_i, hip_i, an_i), vis = pick_side(lm)

        if vis >= MIN_VISIBILITY:
            def pt(i):
                return (lm[i].x * w, lm[i].y * h)

            sh, el, wr, hip, ank = pt(sh_i), pt(el_i), pt(wr_i), pt(hip_i), pt(an_i)

            elbow_ang = angle(sh, el, wr)
            body_ang = angle(sh, hip, ank)

            smooth_elbow += (elbow_ang - smooth_elbow) * SMOOTHING
            smooth_body += (body_ang - smooth_body) * SMOOTHING

            horizontal = abs(sh[1] - ank[1]) < abs(sh[0] - ank[0]) * 0.6

            if phase == "workout":
                if not horizontal:
                    feedback, feedback_ok = "Get into push-up position", False
                else:
                    # ---- Live form check ----
                    if smooth_body < BODY_MIN_ANGLE:
                        line_y = sh[1] + (ank[1] - sh[1]) * (
                            (hip[0] - sh[0]) / ((ank[0] - sh[0]) or 1e-6))
                        msg = "Hips too high!" if hip[1] < line_y else "Hips sagging!"
                        feedback, feedback_ok = msg, False
                        if stage == "down" or smooth_elbow < UP_ANGLE:
                            rep_form_ok = False
                    else:
                        feedback, feedback_ok = "Good form", True

                    # ---- Rep state machine ----
                    if stage == "up" and smooth_elbow < UP_ANGLE - 15:
                        stage = "down"
                        rep_start = now
                        min_elbow_this_rep = smooth_elbow
                        rep_form_ok = smooth_body >= BODY_MIN_ANGLE

                    if stage == "down":
                        min_elbow_this_rep = min(min_elbow_this_rep, smooth_elbow)

                        if smooth_elbow > UP_ANGLE:
                            stage = "up"
                            last_rep_time = now - rep_start
                            deep_enough = min_elbow_this_rep <= DOWN_ANGLE + 10

                            if not deep_enough:
                                feedback, feedback_ok = "Go lower next time", False
                                rep_form_ok = False
                            elif last_rep_time < MIN_REP_TIME:
                                feedback, feedback_ok = "Too fast, slow down", False
                            elif last_rep_time > MAX_REP_TIME:
                                feedback, feedback_ok = "Too slow, keep moving", False

                            if rep_form_ok or not COUNT_ONLY_GOOD_REPS:
                                total_reps += 1
                                set_reps += 1
                                speak(str(set_reps))
                            if not rep_form_ok:
                                # Bad-form rep: only the bad-form counter goes up
                                bad_reps += 1
                                if feedback_ok:
                                    feedback, feedback_ok = "Bad form - rep not counted", False
                                if COUNT_ONLY_GOOD_REPS:
                                    speak("Not counted")
                                    last_warn = now

                            # ---- Set / goal logic ----
                            if set_reps >= REPS_PER_SET:
                                if current_set >= TOTAL_SETS:
                                    phase = "done"
                                    feedback, feedback_ok = "Goal reached!", True
                                    speak("Goal reached. Great workout!")
                                else:
                                    current_set += 1
                                    set_reps = 0
                                    phase = "rest"
                                    rest_start = now
                                    stage = "up"
                                    speak(f"Set complete. Rest for {REST_SECONDS} seconds.")

                    # ---- Spoken form warnings (with cooldown) ----
                    if (phase == "workout" and feedback in SPOKEN_WARNINGS
                            and now - last_warn > WARNING_COOLDOWN):
                        speak(feedback)
                        last_warn = now

                # Visual aids
                cv2.circle(frame, (int(el[0]), int(el[1])), 8, (255, 0, 255), -1)
                put(frame, f"{int(smooth_elbow)}", (int(el[0]) + 10, int(el[1]) - 10),
                    (255, 0, 255), 0.7)
                body_color = (0, 255, 0) if smooth_body >= BODY_MIN_ANGLE else (0, 0, 255)
                cv2.line(frame, (int(sh[0]), int(sh[1])), (int(ank[0]), int(ank[1])),
                         body_color, 2)
        elif phase == "workout":
            feedback, feedback_ok = "Move so your full body is visible", False

    # ----------------------------- HUD -----------------------------
    calories = total_reps * KCAL_PER_PUSHUP_70KG * (BODY_WEIGHT_KG / 70.0)
    elapsed = int(now - workout_start)

    cv2.rectangle(frame, (0, 0), (w, 135), (30, 30, 30), -1)
    put(frame, f"SET {min(current_set, TOTAL_SETS)}/{TOTAL_SETS}", (20, 35), (255, 255, 255), 0.9)
    put(frame, f"REPS: {set_reps}/{REPS_PER_SET}", (20, 85), (0, 255, 255), 1.4, 3)
    put(frame, f"Stage: {stage}", (20, 120), (200, 200, 200), 0.7)

    col2 = w - 380
    put(frame, f"Total: {total_reps}/{REPS_PER_SET * TOTAL_SETS}", (col2, 30), (255, 255, 255), 0.7)
    put(frame, f"Calories: {calories:.1f} kcal", (col2, 55), (0, 200, 255), 0.7)
    put(frame, f"Last rep: {last_rep_time:.1f}s", (col2, 80), (255, 200, 0), 0.7)
    put(frame, f"Bad form reps: {bad_reps}   Time: {elapsed // 60:02d}:{elapsed % 60:02d}",
        (col2, 105), (0, 165, 255), 0.6)
    put(frame, feedback, (col2, 128), (0, 255, 0) if feedback_ok else (0, 0, 255), 0.7)

    # Rest / done overlays
    if phase == "rest":
        remaining = max(0, int(math.ceil(REST_SECONDS - (now - rest_start))))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 135), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
        put(frame, "REST", (w // 2 - 70, h // 2 - 40), (0, 255, 255), 2, 4)
        put(frame, f"{remaining}s", (w // 2 - 50, h // 2 + 30), (255, 255, 255), 2, 4)
        put(frame, f"Next: set {current_set}", (w // 2 - 90, h // 2 + 80), (200, 200, 200), 0.8)
    elif phase == "done":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 135), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
        put(frame, "GOAL REACHED!", (w // 2 - 190, h // 2 - 20), (0, 255, 0), 1.6, 4)
        put(frame, f"{total_reps} reps | {calories:.1f} kcal | {elapsed // 60:02d}:{elapsed % 60:02d}",
            (w // 2 - 190, h // 2 + 30), (255, 255, 255), 0.8)
        put(frame, "Press R to restart or Q to quit", (w // 2 - 190, h // 2 + 70),
            (200, 200, 200), 0.7)

    cv2.imshow("Push-up Counter", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        reset_all()
        speak("Workout reset. Set one starting.")

speech_q.put(None)
cap.release()
cv2.destroyAllWindows()
from flask import Flask, request, jsonify
from ortools.sat.python import cp_model
import os

app = Flask(__name__)

# --------------------------
# Helper Classes
# --------------------------
class Teacher:
    def __init__(self, id, name, subject_ids, unavailable_slots):
        self.id = id
        self.name = name
        self.subject_ids = subject_ids
        self.unavailable_slots = set(unavailable_slots)

class Section:
    def __init__(self, id, name, subject_ids):
        self.id = id
        self.name = name
        self.subject_ids = subject_ids

class Room:
    def __init__(self, id, name, is_lab, unavailable_slots=None):
        self.id = id
        self.name = name
        self.is_lab = is_lab
        self.unavailable_slots = set(unavailable_slots or [])

class Subject:
    def __init__(self, id, name, lec_per_week, requires_lab):
        self.id = id
        self.name = name
        self.lec_per_week = lec_per_week
        self.requires_lab = requires_lab

class LectureSlot:
    def __init__(self, id, day, start_time, end_time):
        self.id = id
        self.day = day
        self.start_time = start_time
        self.end_time = end_time

# --------------------------
# Timetable Generator
# --------------------------
def create_timetable_for_section(section, teachers, rooms, subjects, lecture_slots):
    model = cp_model.CpModel()
    subject_map = {s.id: s for s in subjects}

    # Check for missing subjects
    missing_subjects = [s_id for s_id in section.subject_ids if s_id not in subject_map]
    if missing_subjects:
        return {"error": f"Subjects not found: {missing_subjects}", "status": "fail"}

    assignment = {}
    for slot_idx, _ in enumerate(lecture_slots):
        for subj_id in section.subject_ids:
            subject_obj = subject_map[subj_id]
            for teacher in teachers:
                if subj_id not in teacher.subject_ids:
                    continue
                if slot_idx in teacher.unavailable_slots:
                    continue
                for room in rooms:
                    if subject_obj.requires_lab and not room.is_lab:
                        continue
                    if slot_idx in room.unavailable_slots:
                        continue
                    var_name = f"{section.id}_{slot_idx}_{subj_id}_{teacher.id}_{room.id}"
                    assignment[(slot_idx, subj_id, teacher.id, room.id)] = model.NewBoolVar(var_name)

    # 1. Total lectures per subject
    for subj_id in section.subject_ids:
        required_lec = subject_map[subj_id].lec_per_week
        vars_list = [var for (slot_idx, s_id, t_id, r_id), var in assignment.items() if s_id == subj_id]
        if vars_list:
            model.Add(sum(vars_list) == required_lec)

    # 2. Only one teacher per subject
    for subj_id in section.subject_ids:
        qualified_teachers = [t for t in teachers if subj_id in t.subject_ids]
        teacher_chosen_vars = []
        for teacher in qualified_teachers:
            is_chosen = model.NewBoolVar(f"chosen_{subj_id}_{teacher.id}")
            teacher_chosen_vars.append(is_chosen)
            lecture_vars = [var for (slot_idx, s_id, t_id, r_id), var in assignment.items()
                            if s_id == subj_id and t_id == teacher.id]
            if lecture_vars:
                model.Add(sum(lecture_vars) > 0).OnlyEnforceIf(is_chosen)
                model.Add(sum(lecture_vars) == 0).OnlyEnforceIf(is_chosen.Not())
        if teacher_chosen_vars:
            model.Add(sum(teacher_chosen_vars) == 1)

    # 3. Teacher one lecture per slot
    for slot_idx, _ in enumerate(lecture_slots):
        for teacher in teachers:
            vars_teacher = [var for (s_idx, s_id, t_id, r_id), var in assignment.items() if s_idx == slot_idx and t_id == teacher.id]
            if vars_teacher:
                model.Add(sum(vars_teacher) <= 1)

    # 4. Room one lecture per slot
    for slot_idx, _ in enumerate(lecture_slots):
        for room in rooms:
            vars_room = [var for (s_idx, s_id, t_id, r_id), var in assignment.items() if s_idx == slot_idx and r_id == room.id]
            if vars_room:
                model.Add(sum(vars_room) <= 1)

    # --------------------------
    # Solve
    # --------------------------
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return {"error": "No feasible timetable found", "status": "fail"}

    timetable = []
    for (slot_idx, subj_id, teacher_id, room_id), var in assignment.items():
        if solver.BooleanValue(var):
            slot = lecture_slots[slot_idx]
            teacher = next(t for t in teachers if t.id == teacher_id)
            room = next(r for r in rooms if r.id == room_id)
            teacher.unavailable_slots.add(slot_idx)
            room.unavailable_slots.add(slot_idx)
            timetable.append({
                "section": section.name,
                "subject": subject_map[subj_id].name,
                "teacher": teacher.name,
                "room": room.name,
                "day": slot.day,
                "start_time": slot.start_time,
                "end_time": slot.end_time
            })

    return {"timetable": timetable, "status": "success"}

# --------------------------
# Flask API
# --------------------------
@app.route("/generate_section_timetable", methods=["POST"])
def generate_section_timetable():
    data = request.get_json()
    try:
        teachers = [Teacher(**t) for t in data.get("teachers", [])]
        rooms = [Room(**r) for r in data.get("rooms", [])]
        subjects = [Subject(**s) for s in data.get("subjects", [])]
        lecture_slots = [LectureSlot(**l) for l in data.get("lectureSlots", [])]
        section_data = data.get("section", {})

        if not section_data:
            return jsonify({"error": "Section data missing", "status": "fail"}), 400

        section = Section(**section_data)
        result = create_timetable_for_section(section, teachers, rooms, subjects, lecture_slots)

        status_code = 200 if result.get("status") == "success" else 500
        return jsonify(result), status_code

    except Exception as e:
        return jsonify({"error": str(e), "status": "fail"}), 500

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

from flask import Flask, request, jsonify
from ortools.sat.python import cp_model

app = Flask(__name__)

# --------------------------
# Helper Classes
# --------------------------
class Teacher:
    def __init__(self, id, name, subject_ids, unavailable_slots):
        self.id = id
        self.name = name
        self.subject_ids = subject_ids
        self.unavailable_slots = unavailable_slots  # List of slot indices

class Section:
    def __init__(self, id, name, subject_ids):
        self.id = id
        self.name = name
        self.subject_ids = subject_ids

class Room:
    def __init__(self, id, name, is_lab):
        self.id = id
        self.name = name
        self.is_lab = is_lab

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
def create_timetable(teachers, sections, rooms, subjects, lecture_slots):
    model = cp_model.CpModel()

    teacher_map = {t.id: t for t in teachers}
    section_map = {s.id: s for s in sections}
    room_map = {r.id: r for r in rooms}
    subject_map = {s.id: s for s in subjects}

    # Variables: (section, slot, subject) -> (teacher, room)
    assignment = {}
    for section in sections:
        for slot_idx, slot in enumerate(lecture_slots):
            for subj_id in section.subject_ids:
                for teacher in teachers:
                    for room in rooms:
                        # Skip non-lab rooms if subject requires lab
                        if subject_map[subj_id].requires_lab and not room.is_lab:
                            continue
                        # Skip unavailable slots for teacher
                        if slot_idx in teacher.unavailable_slots:
                            continue
                        var_name = f"{section.id}_{slot_idx}_{subj_id}_{teacher.id}_{room.id}"
                        assignment[(section.id, slot_idx, subj_id, teacher.id, room.id)] = model.NewBoolVar(var_name)

    # --------------------------
    # Constraints
    # --------------------------

    # 1. Each lecture of a subject per section assigned exactly once
    for section in sections:
        for subj_id in section.subject_ids:
            required_lec = subject_map[subj_id].lec_per_week
            vars_list = []
            for slot_idx, slot in enumerate(lecture_slots):
                for teacher in teachers:
                    for room in rooms:
                        key = (section.id, slot_idx, subj_id, teacher.id, room.id)
                        if key in assignment:
                            vars_list.append(assignment[key])
            model.Add(sum(vars_list) == required_lec)

    # 2. One teacher per subject per section
    for section in sections:
        for subj_id in section.subject_ids:
            teacher_vars = []
            for teacher in teachers:
                vars_for_teacher = []
                for slot_idx, slot in enumerate(lecture_slots):
                    for room in rooms:
                        key = (section.id, slot_idx, subj_id, teacher.id, room.id)
                        if key in assignment:
                            vars_for_teacher.append(assignment[key])
                teacher_vars.append(sum(vars_for_teacher))
            # Exactly one teacher assigned to all lectures
            bool_vars = [model.NewBoolVar(f"teacher_assigned_{section.id}_{subj_id}_{t.id}") for t, _ in zip(teachers, teacher_vars)]
            for b, v in zip(bool_vars, teacher_vars):
                model.Add(v > 0).OnlyEnforceIf(b)
                model.Add(v == 0).OnlyEnforceIf(b.Not())
            model.Add(sum(bool_vars) == 1)

    # 3. Teacher can only teach one lecture per section per day
    for section in sections:
        for teacher in teachers:
            # Collect lectures for this section-teacher
            for day in set(slot.day for slot in lecture_slots):
                day_slots = [i for i, s in enumerate(lecture_slots) if s.day == day]
                vars_day = []
                for slot_idx in day_slots:
                    for subj_id in section.subject_ids:
                        for room in rooms:
                            key = (section.id, slot_idx, subj_id, teacher.id, room.id)
                            if key in assignment:
                                vars_day.append(assignment[key])
                if vars_day:
                    model.Add(sum(vars_day) <= 1)

    # 4. Room cannot have more than one lecture at same slot
    for slot_idx, slot in enumerate(lecture_slots):
        for room in rooms:
            vars_room = []
            for section in sections:
                for subj_id in section.subject_ids:
                    for teacher in teachers:
                        key = (section.id, slot_idx, subj_id, teacher.id, room.id)
                        if key in assignment:
                            vars_room.append(assignment[key])
            if vars_room:
                model.Add(sum(vars_room) <= 1)

    # --------------------------
    # Optional: soft constraint to add gaps between teacher lectures
    # --------------------------
    penalty_vars = []
    for teacher in teachers:
        for slot_idx in range(len(lecture_slots) - 1):
            for section1 in sections:
                for section2 in sections:
                    for subj1 in section1.subject_ids:
                        for subj2 in section2.subject_ids:
                            for room1 in rooms:
                                for room2 in rooms:
                                    key1 = (section1.id, slot_idx, subj1, teacher.id, room1.id)
                                    key2 = (section2.id, slot_idx+1, subj2, teacher.id, room2.id)
                                    if key1 in assignment and key2 in assignment:
                                        gap_var = model.NewBoolVar(f"gap_{teacher.id}_{slot_idx}")
                                        model.Add(assignment[key1] + assignment[key2] <= 1).OnlyEnforceIf(gap_var)
                                        penalty_vars.append(gap_var)

    model.Maximize(sum(penalty_vars))

    # --------------------------
    # Solve
    # --------------------------
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        timetable = []
        for section in sections:
            for slot_idx, slot in enumerate(lecture_slots):
                for subj_id in section.subject_ids:
                    for teacher in teachers:
                        for room in rooms:
                            key = (section.id, slot_idx, subj_id, teacher.id, room.id)
                            if key in assignment and solver.BooleanValue(assignment[key]):
                                timetable.append({
                                    "section": section.name,
                                    "subject": subject_map[subj_id].name,
                                    "teacher": teacher.name,
                                    "room": room.name,
                                    "day": slot.day,
                                    "start_time": slot.start_time,
                                    "end_time": slot.end_time
                                })
        return timetable
    else:
        return []

# --------------------------
# Flask API
# --------------------------
@app.route("/generate_timetable", methods=["POST"])
def generate_timetable_api():
    data = request.get_json()
    try:
        teachers = [Teacher(**t) for t in data["teachers"]]
        sections = [Section(**s) for s in data["sections"]]
        rooms = [Room(**r) for r in data["rooms"]]
        subjects = [Subject(**s) for s in data["subjects"]]
        lecture_slots = [LectureSlot(**l) for l in data["lectureSlots"]]

        timetable = create_timetable(teachers, sections, rooms, subjects, lecture_slots)
        return jsonify({"timetable": timetable, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "fail"}), 500

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)

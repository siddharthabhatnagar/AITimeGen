from flask import Flask, request, jsonify
from ortools.sat.python import cp_model
import json

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
# Utility Functions
# --------------------------

def create_timetable(teachers, sections, rooms, subjects, lecture_slots):
    model = cp_model.CpModel()

    # Map IDs to objects for convenience
    teacher_map = {t.id: t for t in teachers}
    section_map = {s.id: s for s in sections}
    room_map = {r.id: r for r in rooms}
    subject_map = {s.id: s for s in subjects}

    # Variables: For each section, slot, subject -> assign teacher and room
    assignment = {}
    for section in sections:
        for slot_idx, slot in enumerate(lecture_slots):
            for subj_id in section.subject_ids:
                for room in rooms:
                    for teacher in teachers:
                        var_name = f"{section.id}_{slot_idx}_{subj_id}_{teacher.id}_{room.id}"
                        assignment[(section.id, slot_idx, subj_id, teacher.id, room.id)] = model.NewBoolVar(var_name)

    # --------------------------
    # Constraints
    # --------------------------

    # 1. Each lecture of a subject for a section assigned exactly once
    for section in sections:
        for subj_id in section.subject_ids:
            required_lec = subject_map[subj_id].lec_per_week
            all_vars = []
            for slot_idx, slot in enumerate(lecture_slots):
                for teacher in teachers:
                    for room in rooms:
                        all_vars.append(assignment[(section.id, slot_idx, subj_id, teacher.id, room.id)])
            model.Add(sum(all_vars) == required_lec)

    # 2. One teacher cannot be in two places at same slot
    for slot_idx, slot in enumerate(lecture_slots):
        for teacher in teachers:
            vars_for_teacher = []
            for section in sections:
                for subj_id in section.subject_ids:
                    for room in rooms:
                        vars_for_teacher.append(assignment[(section.id, slot_idx, subj_id, teacher.id, room.id)])
            model.Add(sum(vars_for_teacher) <= 1)

    # 3. One room cannot be assigned twice at same slot
    for slot_idx, slot in enumerate(lecture_slots):
        for room in rooms:
            vars_for_room = []
            for section in sections:
                for subj_id in section.subject_ids:
                    for teacher in teachers:
                        vars_for_room.append(assignment[(section.id, slot_idx, subj_id, teacher.id, room.id)])
            model.Add(sum(vars_for_room) <= 1)

    # 4. Teacher unavailable slots
    for teacher in teachers:
        for slot_idx in teacher.unavailable_slots:
            for section in sections:
                for subj_id in section.subject_ids:
                    for room in rooms:
                        model.Add(assignment[(section.id, slot_idx, subj_id, teacher.id, room.id)] == 0)

    # 5. One teacher should take all lectures of one subject for one section
    for section in sections:
        for subj_id in section.subject_ids:
            teacher_vars_per_teacher = []
            for teacher in teachers:
                vars_for_teacher = []
                for slot_idx, slot in enumerate(lecture_slots):
                    for room in rooms:
                        vars_for_teacher.append(assignment[(section.id, slot_idx, subj_id, teacher.id, room.id)])
                teacher_vars_per_teacher.append(sum(vars_for_teacher))
            # Exactly one teacher assigned to all lectures
            model.Add(sum([v > 0 for v in teacher_vars_per_teacher]) == 1)

    # 6. Teacher should have at least one gap between lectures
    # (Soft constraint: try to minimize consecutive lectures)
    # We'll define a penalty
    penalty_vars = []
    for teacher in teachers:
        for section in sections:
            for subj_id in section.subject_ids:
                for slot_idx in range(len(lecture_slots)-1):
                    for room1 in rooms:
                        for room2 in rooms:
                            consecutive = model.NewBoolVar(f"gap_{teacher.id}_{section.id}_{subj_id}_{slot_idx}")
                            model.Add(assignment[(section.id, slot_idx, subj_id, teacher.id, room1.id)] +
                                      assignment[(section.id, slot_idx+1, subj_id, teacher.id, room2.id)] <= 1).OnlyEnforceIf(consecutive)
                            penalty_vars.append(consecutive)

    # Objective: maximize gaps (soft constraint)
    model.Maximize(sum(penalty_vars))

    # --------------------------
    # Solve
    # --------------------------
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        timetable = []
        for section in sections:
            for slot_idx, slot in enumerate(lecture_slots):
                for subj_id in section.subject_ids:
                    for teacher in teachers:
                        for room in rooms:
                            var = assignment[(section.id, slot_idx, subj_id, teacher.id, room.id)]
                            if solver.BooleanValue(var):
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
def generate_timetable():
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

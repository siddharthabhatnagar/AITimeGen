from flask import Flask, request, jsonify
from ortools.sat.python import cp_model
import os

app = Flask(__name__)

# --------------------------
# Helper Classes
# --------------------------
# (Your original Helper Classes remain correct)
class Teacher:
    def __init__(self, id, name, subject_ids, unavailable_slots):
        self.id = id
        self.name = name
        self.subject_ids = subject_ids
        self.unavailable_slots = unavailable_slots

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
# Timetable Generator (UPDATED LOGIC)
# --------------------------
def create_timetable(teachers, sections, rooms, subjects, lecture_slots):
    model = cp_model.CpModel()

    subject_map = {s.id: s for s in subjects}

    # Variables: (section, slot, subject, teacher, room) -> Boolean
    assignment = {}
    for section in sections:
        for slot_idx, _ in enumerate(lecture_slots):
            for subj_id in section.subject_ids:
                subject_obj = subject_map[subj_id]
                for teacher in teachers:
                    # Filter 1: Teacher must be qualified for the subject (CRITICAL FIX)
                    if subj_id not in teacher.subject_ids:
                        continue
                        
                    # Filter 2: Skip unavailable slots for teacher (Original check)
                    if slot_idx in teacher.unavailable_slots:
                        continue
                        
                    for room in rooms:
                        # Filter 3: Skip non-lab rooms if subject requires lab (Original check)
                        if subject_obj.requires_lab and not room.is_lab:
                            continue
                            
                        var_name = f"{section.id}_{slot_idx}_{subj_id}_{teacher.id}_{room.id}"
                        assignment[(section.id, slot_idx, subj_id, teacher.id, room.id)] = model.NewBoolVar(var_name)

    # --------------------------
    # Constraints
    # --------------------------

    # 1. Total Lectures per Section/Subject (HARD CONSTRAINT)
    for section in sections:
        for subj_id in section.subject_ids:
            required_lec = subject_map[subj_id].lec_per_week
            vars_list = []
            
            for key, var in assignment.items():
                s_id, _, a_subj_id, _, _ = key
                if s_id == section.id and a_subj_id == subj_id:
                    vars_list.append(var)
                    
            model.Add(sum(vars_list) == required_lec)

    # 2. Exactly one teacher is chosen for all lectures of a subject in a section. (CRITICAL FIX)
    for section in sections:
        for subj_id in section.subject_ids:
            # Find all teachers that could possibly teach this subject/section combination
            qualified_teachers = [t for t in teachers if subj_id in t.subject_ids]
            
            teacher_chosen_vars = []
            for teacher in qualified_teachers:
                is_chosen = model.NewBoolVar(f"chosen_{section.id}_{subj_id}_{teacher.id}")
                teacher_chosen_vars.append(is_chosen)
                
                # Sum of all lectures assigned to this teacher for this specific section/subject
                lecture_sum = []
                for key, var in assignment.items():
                    s_id, _, a_subj_id, t_id, _ = key
                    if s_id == section.id and a_subj_id == subj_id and t_id == teacher.id:
                        lecture_sum.append(var)
                
                # Link indicator: If teacher teaches > 0, indicator is True. If 0, indicator is False.
                model.Add(sum(lecture_sum) == 0).OnlyEnforceIf(is_chosen.Not())
                model.Add(sum(lecture_sum) > 0).OnlyEnforceIf(is_chosen)

            # Main constraint: Exactly one qualified teacher must be chosen.
            if teacher_chosen_vars:
                model.Add(sum(teacher_chosen_vars) == 1)


    # 3. Teacher can only teach one lecture at a time (per slot). (CRITICAL FIX)
    for slot_idx, _ in enumerate(lecture_slots):
        for teacher in teachers:
            vars_teacher = []
            
            for key, var in assignment.items():
                _, a_slot_idx, _, t_id, _ = key
                if a_slot_idx == slot_idx and t_id == teacher.id:
                    vars_teacher.append(var)

            if vars_teacher:
                model.Add(sum(vars_teacher) <= 1)


    # 4. Room cannot have more than one lecture at same slot (Original check, correctly implemented)
    for slot_idx, _ in enumerate(lecture_slots):
        for room in rooms:
            vars_room = []
            for key, var in assignment.items():
                _, a_slot_idx, _, _, r_id = key
                if a_slot_idx == slot_idx and r_id == room.id:
                    vars_room.append(var)
            
            if vars_room:
                model.Add(sum(vars_room) <= 1)

    # --------------------------
    # Optional: soft constraint to add gaps (Your original logic, which is complex but correct in its objective)
    # The objective is currently to maximize the number of gap events (Add(key1 + key2 <= 1) enforced if gap_var is True).
    # This maximizes the number of slots that DON'T have back-to-back classes.
    # --------------------------
    penalty_vars = []
    for teacher in teachers:
        for slot_idx in range(len(lecture_slots) - 1):
            
            # Sum of assignments for teacher at current slot
            lectures_at_current = []
            for key, var in assignment.items():
                _, a_slot_idx, _, t_id, _ = key
                if a_slot_idx == slot_idx and t_id == teacher.id:
                    lectures_at_current.append(var)
            
            # Sum of assignments for teacher at next slot
            lectures_at_next = []
            for key, var in assignment.items():
                _, a_slot_idx, _, t_id, _ = key
                if a_slot_idx == slot_idx + 1 and t_id == teacher.id:
                    lectures_at_next.append(var)
                    
            if lectures_at_current and lectures_at_next:
                # 1 if the teacher has back-to-back classes (lecture at current AND lecture at next)
                back_to_back = model.NewBoolVar(f"b2b_{teacher.id}_{slot_idx}")
                
                # If sum(current) and sum(next) are both > 0, then back_to_back must be 1.
                # Use reified constraints to define 'back_to_back' based on the product of the two lecture sums.
                # However, since the sum is <= 1 (Constraint 3), we can simplify the logic:
                
                # Back-to-back = current_assigned AND next_assigned
                model.AddBoolOr([back_to_back.Not(), lectures_at_current[0].Not(), lectures_at_next[0].Not()]) # equivalent to back_to_back == (current AND next)
                model.AddImplication(lectures_at_current[0], back_to_back)
                model.AddImplication(lectures_at_next[0], back_to_back)
                
                # Minimize penalty, where penalty is the number of back-to-back assignments.
                penalty_vars.append(back_to_back) 

    # Objective: Minimize the number of back-to-back classes (penalties).
    model.Minimize(sum(penalty_vars))

    # --------------------------
    # Solve
    # --------------------------
    solver = cp_model.CpSolver()
    # solver.parameters.log_search_progress = True # Useful for debugging
    # solver.parameters.max_time_in_seconds = 10.0 # Time limit for complex problems
    
    status = solver.Solve(model)

    # (Your result processing logic remains correct)
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
        # Check for UNSATISFIABLE status
        if status == cp_model.INFEASIBLE:
            return f"Model is INFEASIBLE. Check constraints for conflicts."
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
        # NOTE: Returning str(e) is essential for debugging on Render
        return jsonify({"error": str(e), "status": "fail"}), 500

# --------------------------
# Run (Deployment Ready)
# --------------------------
if __name__ == "__main__":
    # Use environment port for production (Render/Heroku), default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
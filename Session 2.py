def get_student_details():
    name = input("Enter student's name: ")
    roll_no = input("Enter roll number: ")
    return {'name': name, 'roll_no': roll_no, 'scores': {}}


def input_scores(student, subjects):
    for subject in subjects:
        score = float(input(f"Enter marks for {subject}: "))
        student['scores'][subject] = score


def calculate_average(student):
    total = sum(student['scores'].values())
    return total / len(student['scores'])


def assign_grade(avg):
    if avg >= 90:
        return 'A+'
    elif avg >= 80:
        return 'A'
    elif avg >= 70:
        return 'B'
    elif avg >= 60:
        return 'C'
    elif avg >= 50:
        return 'D'
    else:
        return 'F'


def find_subject_toppers(students, subjects):
    toppers = {subject: [] for subject in subjects}
    top_scores = {subject: 0 for subject in subjects}

    for student in students:
        for subject in subjects:
            score = student['scores'][subject]
            if score > top_scores[subject]:
                top_scores[subject] = score
                toppers[subject] = [student['name']]
            elif score == top_scores[subject]:
                toppers[subject].append(student['name'])
    return toppers, top_scores


def highlight_common_top_scores(top_scores):
    # Reverse map: score -> list of subjects
    score_subject_map = {}
    for subject, score in top_scores.items():
        score_subject_map.setdefault(score, []).append(subject)

    # Subjects with same top score
    patterns = {score: tuple(subjects) for score, subjects in score_subject_map.items() if len(subjects) > 1}
    return patterns


def display_report(student, avg, grade):
    print(f"\nReport for {student['name']} (Roll No: {student['roll_no']})")
    for subject, score in student['scores'].items():
        print(f"{subject}: {score}")
    print(f"Average: {avg:.2f}")
    print(f"Grade: {grade}")


def main():
    subjects = ['Math', 'Science', 'English', 'History', 'Computer']
    num_students = int(input("Enter number of students: "))
    students = []

    for _ in range(num_students):
        student = get_student_details()
        input_scores(student, subjects)
        students.append(student)

    # Process each student
    for student in students:
        avg = calculate_average(student)
        grade = assign_grade(avg)
        student['average'] = avg
        student['grade'] = grade
        display_report(student, avg, grade)

    # Subject-wise toppers
    toppers, top_scores = find_subject_toppers(students, subjects)
    print("\nSubject-wise Toppers:")
    for subject, names in toppers.items():
        print(f"{subject}: {', '.join(names)} with score {top_scores[subject]}")

    # Highlight subjects with same top scores
    common_patterns = highlight_common_top_scores(top_scores)
    if common_patterns:
        print("\nSubjects with same top scores:")
        for score, subject_tuple in common_patterns.items():
            print(f"Score {score} in subjects: {', '.join(subject_tuple)}")
    else:
        print("\nNo subjects with identical top scores.")


if __name__ == "__main__":
    main()

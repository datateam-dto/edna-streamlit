# QA of Lesson Slides: Suggested Prompts and Prompt Flows

The quality assurance (QA) of lesson slides can significantly benefit from a structured approach using AI. Below are suggested prompts and prompt flows designed to evaluate and improve the quality of educational materials based on various criteria. These suggestions are organized to address different aspects of lesson quality, such as teaching strategies, alignment with objectives, inclusion of essential components, question and instruction extraction, and quiz creation.

## Teaching Strategies Evaluation

**Prompt 1:**
```
Evaluate the teaching strategies utilized in the following slides text:

{input_text}.

//end of slides text.

The teaching strategies to assess include:

1) Experiential and Situated Learning
2) Reflective Learning
3) Constructivism
4) Collaborative Learning
5) Guided Instruction (Incorporating Socio-Cultural Theory and Developmental Stages of Learning)
6) Discovery Learning or Inquiry-Based Learning
7) Direct Instruction
8) Concrete-Pictorial-Abstract Approach
9) George Polya's Four-Step Problem-Solving Method
10) Socratic Questioning

Provide an analysis on the application of these strategies within the given text. Output JSON with teaching strategies for keys and feedback regarding the application of the teaching strategies in the slides text as the values.
```

## Alignment with Objectives and Engagement Strategies

**Prompt 2:**
```
Analyze {input_text} for its alignment with SMART objectives, DepEd competencies, and engagement strategies. Ensure the analysis checks for varied assessments, differentiated instruction, ICT integration, and support for diverse learners, all within a coherent flow. Provide feedback concisely in one short paragraph without summarizing or rephrasing the lesson plan.
```

## Essential Components Inclusion

**Prompt 3:**
```
Review the following slides text to ensure inclusion of essential components:

List of essential components:
1. Title Slide ↔ Content: Accurate representation of the specific subject and lesson.
2. Objectives ↔ Objectives: Clear goals for student achievements by lesson’s end.
3. Motivation Warm Up and Review ↔ Procedure: Activation of prior knowledge and connection to new lesson.
4. Transition Slides ↔ Procedure: Seamless progression into main lesson content.
5. Main Lesson Presentation ↔ Procedure: Thematic presentation fostering student engagement and skill development.
6. Sample Practice ↔ Procedure: Thematic activities for skill application.
7. Practice Exercises ↔ Procedure: Collaborative practice of learned skills.
8. Independent Work ↔ Procedure: Tasks reinforcing mastery and application.
9. Application ↔ Procedure: Discussion on real-world skill relevance.
Generalization ↔ Procedure: Summary of key points.
Evaluation ↔ Procedure: Assessments aligned with objectives.
Exit Activity ↔ Remarks, Reflection: Lesson conclusion promoting reflection.

Slides text:
{input_text}

Output JSON showing whether the component exists in the slides text.
```

## Questions and Instructions Extraction

**Prompt 4:**
```
Review the following slides text to extract questions and instructions that may be under a component:

List of essential components:
1. Title Slide ↔ Content
2. Objectives ↔ Objectives
3. Motivation Warm Up and Review ↔ Procedure
4. Transition Slides ↔ Procedure
5. Main Lesson Presentation ↔ Procedure
6. Sample Practice ↔ Procedure
7. Practice Exercises ↔ Procedure
8. Independent Work ↔ Procedure
9. Application ↔ Procedure
Generalization ↔ Procedure
Evaluation ↔ Procedure
Exit Activity ↔ Remarks

Slides text:
{input_text}

Just provide output with components as the keys and a nested list with questions, instructions as keys with the relevant values.

e.g.
1. Component
• Questions:
• Instructions:

2. Component
• Questions:
• Instructions:
```

## Quiz Creation Based on Content

**Prompt 5:**
```
Design a 10-item quiz based on the content of {input_text}, including brief explanations for both correct and incorrect answers.
```

## Simplified Content Summarization

**Chain Output 2:**
```
Offer a brief, one-paragraph summary of {input_text} for a concise encapsulation.
```

These prompts are designed to be used in sequence or individually, depending on the specific needs of the QA process. They can help educators and content creators ensure that their lesson slides are not only engaging and informative but also aligned with educational standards and objectives.
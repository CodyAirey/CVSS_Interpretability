# User Study Survey Instrument

This document reproduces the participant-facing survey instrument used in the user study.

---

## Welcome

### Evaluating the Interpretability and Clarity of CVE Descriptions for CVSS Scoring

Thank you for taking part in this survey.

We are looking for participants who are over 18 years old and who have some involvement with CVEs, whether that is writing, reviewing, or working with them in your role. Your input helps us understand how practitioners interpret and evaluate CVE descriptions when assigning CVSS metrics.

The survey takes approximately 30 minutes. There are no right or wrong answers. We are interested in your reasoning and judgement, not testing your knowledge.

---

## Survey Structure

1. **Information and consent**  
   Please read and confirm before beginning.

2. **Preliminary and demographics**  
   A few questions about your role and background.

3. **Justification tasks**  
   Read CVE descriptions and judge how well they support a specific CVSS value.

4. **Feedback section**  
   Space for additional comments.

---

## Participant Information and Consent

Participants were shown:

- Participant information sheet  
- Participant consent form  

By selecting **Begin**, participants confirmed that they had read and understood both documents and agreed to take part.

---

## Before You Begin

### Role in Relation to CVEs

Which of the following best describes your role?

- I work for a CVE Numbering Authority (CNA) and write or approve CVE descriptions.
- I use CVE records in my professional work but do not author them.
- I both use CVE records and write or approve their descriptions as part of a CNA.
- I work for an organisation that produces vulnerability advisories (but is not a CNA).
- Other (free text)
- I am not involved with CVEs in any way.

### Age Confirmation

- Yes, I am 18 or older
- No

---

## Demographic Information

### Age

- 18–24  
- 25–34  
- 35–44  
- 45–54  
- 55–64  
- 65 or older  

### Highest Level of Education

- High school or equivalent  
- Vocational training or certificate  
- Bachelor's degree  
- Master's degree  
- Doctorate  
- Other (free text)

### Is English Your First Language?

- Yes  
- No  

### Confidence in Reading or Writing Technical Security Documentation in English

- Not confident  
- Somewhat confident  
- Neutral  
- Confident  
- Very confident  

### Current Role (Select All That Apply)

- Security analyst or researcher  
- Software developer or engineer  
- Vulnerability coordinator or CNA  
- Academic  
- Student  
- Penetration tester / Red team / Bug bounty hunter  
- Infrastructure or operations (DevOps, sysadmin)  
- Policy, compliance, or governance  
- Other (free text)

### Familiarity with Common Security Vulnerabilities and Defences

- I do not know many named security concepts or defences  
- I know some names and have a rough idea what they are  
- I know many of these terms and feel comfortable using them in context  

### Interaction with CVE Records

How often do you interact with CVE records?

- Daily  
- Weekly  
- Monthly  
- Yearly or less frequently  

### Writing or Overseeing CVE Descriptions

- I do not  
- Daily  
- Weekly  
- Monthly  
- A few times per year  
- Less than once a year  

### Experience with CVSS Scoring

Have you ever assigned or contributed to a CVSS score?

- No  
- Yes, informally  
- Yes, formally  

### CVSS Versions Familiar With (Select All That Apply)

- CVSS 2.0  
- CVSS 3.0  
- CVSS 3.1  
- CVSS 4.0  
- None / Not sure  

### Formal Training or Certification in CVSS (Select All That Apply)

- CVSS 2.0  
- CVSS 3.0  
- CVSS 3.1  
- CVSS 4.0  
- None / Not sure  

---

# Justification Tasks

## CVE Evaluation: Privileges Required (CVSS v3.1)

Participants evaluated whether each CVE description justified the recorded ground truth CVSS v3.1 Privileges Required (PR) value when read against the official specification.

Reference:  
https://www.first.org/cvss/v3-1/specification-document

For each CVE, participants were shown:

- The CVE identifier and description text  
- The recorded ground truth PR value  
- The PR specification excerpt  
- A single-response Justification Value (JV) judgement on an 8-point scale  

---

## Per-Item Layout

Each participant completed ten CVE evaluations.

### CVE Description

```
[CVE DESCRIPTION HERE]
```

### Ground Truth

```
[GROUND TRUTH PR VALUE HERE]
```

---

## Primary Question

**Based on the CVE description alone (no external knowledge), to what extent does the description justify the assigned Privileges Required value?**

- Strongly contradicts - explicitly implies a different value  
- Contradicts - more consistent with a different value  
- Leans against - weak cues point away from the target  
- No evidence - cannot infer from the text  
- Leans toward - partial or indirect support  
- Supports - clear cues imply the target  
- Explicitly supports - directly states or uniquely implies the target  
- Unclear - text is poorly written or cannot be understood  

---

## Follow-Up Statements (Likert Scale)

Participants rated the following on a five-point scale from Strongly Disagree to Strongly Agree:

- The description explicitly mentions privilege requirements.  
- The description allows privilege requirements to be inferred from context.  
- The description provides sufficient information without relying on outside knowledge.  
- The wording is vague or unclear regarding privilege requirements.  
- Different parts of the description suggest different privilege levels.  

---

## Optional Free-Text Responses

Participants could optionally provide:

- External resources consulted  
- Additional comments about the justification  

---

## Repetition and Sampling

Each participant evaluated ten CVEs.

The ten CVEs were sampled from a prepared pool of sixty candidate CVEs.

The full pool and associated materials are linked in the thesis artefacts.

All per-participant CVE assignments and responses are included in the released CSV accompanying the thesis.

---

# Final Reflections

Participants rated the following on a five-point scale from Very Low to Very High:

- Clarity of instructions  
- Ease of deciding on answers  
- Confidence in answers  
- Mental fatigue  

---

## Final Feedback

Participants could provide open-ended feedback about the survey.

If they wished to be notified of the results, they could optionally provide an email address.

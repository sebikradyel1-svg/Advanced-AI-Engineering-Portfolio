#!/usr/bin/env python3
"""
English Legal/Business LLM Fine-Tuning Script - LONG QUALITY TRAINING
======================================================================

Optimized for extended training to achieve best quality results.
Includes evaluation, TensorBoard logging, and checkpoint management.

RECOMMENDED CONFIGURATIONS:

    # QUALITY TRAINING (2-3 hours on RTX 3060) - RECOMANDAT
    python english_legal_llm_finetuning_LONG.py --preset quality
    
    # EXTENDED TRAINING (4-5 hours) - Maximum quality
    python english_legal_llm_finetuning_LONG.py --preset extended
    
    # OVERNIGHT TRAINING (6-8 hours) - Best possible
    python english_legal_llm_finetuning_LONG.py --preset overnight

    # Custom configuration
    python english_legal_llm_finetuning_LONG.py --epochs 15 --lora_r 64 --learning_rate 5e-5

Author: Sebastian Manolache - AI Engineering Portfolio
License: MIT
"""

import argparse
import logging
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# TRAINING PRESETS
# ============================================================================

TRAINING_PRESETS = {
    "quick": {
        "description": "Quick test (30 min)",
        "epochs": 3,
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "batch_size": 2,
        "gradient_accumulation": 4,
        "warmup_ratio": 0.1,
        "eval_split": 0.1,
    },
    "quality": {
        "description": "Quality training (2-3 hours) - RECOMMENDED",
        "epochs": 10,
        "lora_r": 32,
        "lora_alpha": 64,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "warmup_ratio": 0.1,
        "eval_split": 0.1,
    },
    "extended": {
        "description": "Extended training (4-5 hours)",
        "epochs": 15,
        "lora_r": 64,
        "lora_alpha": 128,
        "learning_rate": 5e-5,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "warmup_ratio": 0.15,
        "eval_split": 0.1,
    },
    "overnight": {
        "description": "Overnight training (6-8 hours) - Maximum quality",
        "epochs": 20,
        "lora_r": 64,
        "lora_alpha": 128,
        "learning_rate": 3e-5,
        "batch_size": 1,
        "gradient_accumulation": 16,
        "warmup_ratio": 0.2,
        "eval_split": 0.15,
    },
}


# ============================================================================
# DATASET: English Legal/Business Text Templates (200+ examples)
# ============================================================================

def create_legal_business_dataset() -> List[Dict[str, str]]:
    """
    Creates a comprehensive dataset of English legal and business text templates.
    Contains 200+ diverse examples across multiple categories.
    """
    
    dataset = []
    
    # -------------------------------------------------------------------------
    # 1. EMPLOYMENT CONTRACTS (25 variations)
    # -------------------------------------------------------------------------
    employment_templates = [
        {"instruction": "Write an employment contract introduction for a software engineer position at a tech startup.",
         "output": """EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is entered into as of [DATE], by and between TechVentures Inc., a Delaware corporation ("Company"), and [EMPLOYEE NAME] ("Employee").

WHEREAS, the Company desires to employ Employee as a Software Engineer, and Employee desires to accept such employment, subject to the terms and conditions set forth herein;

NOW, THEREFORE, in consideration of the mutual covenants and agreements contained herein, the parties agree as follows:

1. POSITION AND DUTIES
Employee shall serve as Software Engineer, reporting to the Engineering Manager. Employee shall perform all duties customarily associated with such position, including software development, code review, system architecture design, and technical documentation."""},
        
        {"instruction": "Draft the compensation section of an employment contract for a marketing manager.",
         "output": """COMPENSATION AND BENEFITS

1. Base Salary
Employee shall receive an annual base salary of [AMOUNT] USD, payable in accordance with the Company's standard payroll practices, less applicable withholdings.

2. Performance Bonus
Employee shall be eligible for an annual performance bonus of up to [PERCENTAGE]% of base salary, based on achievement of performance objectives.

3. Benefits
Employee shall participate in all employee benefit plans including:
   a) Health insurance (medical, dental, vision)
   b) 401(k) retirement plan with company matching
   c) Paid time off (PTO) of [NUMBER] days per year
   d) Life and disability insurance

4. Equity Compensation
Subject to Board approval, Employee shall be granted stock options to purchase [NUMBER] shares under the Company's Stock Incentive Plan."""},
        
        {"instruction": "Write the termination clause for an at-will employment agreement.",
         "output": """TERMINATION OF EMPLOYMENT

1. At-Will Employment
Employee's employment with the Company is "at-will," meaning either party may terminate the employment relationship at any time, with or without cause, subject to this Agreement.

2. Termination by Company for Cause
The Company may terminate Employee's employment immediately for Cause, including: (a) material breach of this Agreement; (b) willful misconduct or gross negligence; (c) conviction of any felony; (d) fraud or misappropriation of Company assets.

3. Termination Without Cause
Either party may terminate without Cause upon [NUMBER] days' written notice. Upon termination without Cause by the Company, Employee shall receive severance equal to [NUMBER] months' base salary, subject to execution of a release agreement.

4. Return of Property
Upon termination, Employee shall return all Company property, documents, and confidential information."""},
        
        {"instruction": "Create a confidentiality clause for an employment contract in the financial services industry.",
         "output": """CONFIDENTIALITY AND PROPRIETARY INFORMATION

1. Definition of Confidential Information
"Confidential Information" means all non-public information relating to the Company's business, including: trade secrets, customer lists, client financial data, investment strategies, proprietary algorithms, risk models, pricing information, and business plans.

2. Non-Disclosure Obligations
Employee agrees to: (a) hold all Confidential Information in strict confidence; (b) not disclose to any third party without prior written consent; (c) use Confidential Information solely for employment duties; (d) take all reasonable precautions to prevent unauthorized disclosure.

3. Duration
These confidentiality obligations shall survive termination of employment for [NUMBER] years, or indefinitely for trade secrets.

4. Return of Materials
Upon termination, Employee shall immediately return all materials containing Confidential Information and certify no copies have been retained."""},
        
        {"instruction": "Write an intellectual property assignment clause for a software development employment contract.",
         "output": """INTELLECTUAL PROPERTY AND INVENTIONS

1. Work Product Ownership
All Work Product created by Employee in the course of employment shall be the sole property of the Company. "Work Product" includes all inventions, discoveries, developments, code, algorithms, documentation, and other materials.

2. Assignment of Rights
Employee hereby irrevocably assigns to the Company all right, title, and interest in any Work Product, including all patent rights, copyrights, trade secrets, and intellectual property rights worldwide.

3. Work Made for Hire
To the extent any Work Product constitutes a "work made for hire" under copyright law, the Company is deemed the author.

4. Prior Inventions
Employee has disclosed in Exhibit A all inventions owned by Employee prior to employment that Employee wishes to exclude from this Agreement.

5. Cooperation
Employee agrees to execute all documents reasonably requested to perfect or enforce the Company's rights in any Work Product."""},
        
        {"instruction": "Draft a non-compete clause for a senior executive employment agreement.",
         "output": """NON-COMPETITION AND NON-SOLICITATION

1. Non-Competition
During employment and for [NUMBER] months following termination, Employee shall not engage in or contribute to any Competitive Business within the Restricted Territory.

"Competitive Business" means any business that develops, manufactures, or sells products substantially similar to those of the Company.

2. Non-Solicitation of Employees
For [NUMBER] months following termination, Employee shall not solicit or encourage any employee to leave the Company.

3. Non-Solicitation of Clients
For [NUMBER] months following termination, Employee shall not solicit business from any client with whom Employee had material contact during the final [NUMBER] months of employment.

4. Reasonableness
Employee acknowledges these restrictions are reasonable and necessary to protect legitimate business interests.

5. Enforcement
If any restriction is unenforceable, it shall be modified to the minimum extent necessary to make it enforceable."""},
        
        {"instruction": "Write a remote work policy addendum for an employment contract.",
         "output": """REMOTE WORK POLICY ADDENDUM

1. Remote Work Authorization
Employee is authorized to perform work remotely from their designated home office, subject to this Addendum and Company discretion.

2. Work Schedule
Employee shall maintain regular working hours and be available for meetings and collaboration during core business hours.

3. Workspace Requirements
Employee shall maintain a dedicated workspace ensuring: (a) privacy for confidential communications; (b) reliable internet connectivity; (c) safety and ergonomic standards.

4. Equipment and Expenses
The Company shall provide necessary equipment and a monthly stipend of $[AMOUNT] for home office expenses.

5. Data Security
Employee shall: (a) use only Company-approved devices; (b) connect via VPN; (c) maintain physical security of devices; (d) report security incidents immediately.

6. Revocation
The Company may revoke remote work authorization at any time with reasonable notice."""},
        
        {"instruction": "Draft probationary period terms for a new employee contract.",
         "output": """PROBATIONARY PERIOD

1. Duration
Employee shall serve a probationary period of [NUMBER] months commencing on the Start Date.

2. Purpose
The probationary period allows both parties to assess suitability for continued employment.

3. Performance Reviews
Employee shall receive performance feedback at [30/60/90] days intervals during the probationary period.

4. Termination During Probation
During the probationary period, either party may terminate employment with [NUMBER] days' written notice, or immediately for cause.

5. Benefits During Probation
Employee shall be entitled to [SPECIFY BENEFITS] during the probationary period. Full benefits shall commence upon successful completion.

6. Successful Completion
Upon satisfactory completion of the probationary period, Employee shall be confirmed as a regular employee with all associated benefits and rights."""},
        
        {"instruction": "Write overtime and working hours provisions for an employment contract.",
         "output": """WORKING HOURS AND OVERTIME

1. Standard Hours
Employee's standard working hours shall be [NUMBER] hours per week, typically Monday through Friday, [START TIME] to [END TIME].

2. Overtime Eligibility
[For exempt employees: Employee is classified as exempt and is not entitled to overtime compensation.]
[For non-exempt employees: Employee shall receive overtime pay at 1.5 times the regular hourly rate for hours worked in excess of 40 per week.]

3. Prior Authorization
All overtime must be approved in advance by Employee's supervisor. Unauthorized overtime may result in disciplinary action.

4. Time Recording
Employee shall accurately record all hours worked using the Company's timekeeping system.

5. Flexible Working
Subject to manager approval, Employee may request flexible working arrangements in accordance with Company policy.

6. On-Call Requirements
If applicable, Employee may be required to be available on-call as specified in their job description."""},
        
        {"instruction": "Create an employee code of conduct clause for an employment agreement.",
         "output": """CODE OF CONDUCT

1. Professional Standards
Employee shall conduct themselves with professionalism, integrity, and respect for colleagues, clients, and business partners at all times.

2. Compliance with Laws
Employee shall comply with all applicable federal, state, and local laws, as well as industry regulations relevant to their position.

3. Conflict of Interest
Employee shall avoid any activity, investment, or interest that conflicts with the Company's interests. Any potential conflicts must be disclosed immediately.

4. Use of Company Resources
Company resources, including equipment, supplies, and time, shall be used only for legitimate business purposes.

5. Harassment and Discrimination
Employee shall not engage in harassment, discrimination, or any behavior that creates a hostile work environment.

6. Substance Abuse
Employee shall not report to work under the influence of alcohol or illegal substances and shall comply with the Company's drug-free workplace policy.

7. Reporting Violations
Employee shall report any known or suspected violations of this Code, company policies, or applicable laws through appropriate channels."""},
        
        {"instruction": "Draft a garden leave clause for an executive employment contract.",
         "output": """GARDEN LEAVE

1. Definition
"Garden Leave" means a period during which Employee remains employed but is not required to attend the workplace or perform duties.

2. Company's Right
Upon Employee's notice of resignation or upon notice of termination by the Company, the Company may place Employee on Garden Leave for up to [NUMBER] weeks.

3. Terms During Garden Leave
During Garden Leave:
   a) Employee shall remain employed and bound by all contractual obligations
   b) Employee shall receive full salary and benefits
   c) Employee shall not attend the workplace without prior approval
   d) Employee shall remain available for consultation if reasonably required
   e) Employee shall not contact clients or employees without approval

4. Return of Property
Employee shall return all Company property at the commencement of Garden Leave.

5. Restrictions
Employee shall not commence alternative employment during Garden Leave without written consent."""},
        
        {"instruction": "Write a dispute resolution clause for an employment contract.",
         "output": """DISPUTE RESOLUTION

1. Informal Resolution
The parties shall first attempt to resolve any dispute arising from this Agreement through good faith negotiation between Employee and Employee's manager or HR representative.

2. Mediation
If informal resolution is unsuccessful within [NUMBER] days, either party may request mediation. The parties shall jointly select a mediator and share costs equally.

3. Arbitration
Any dispute not resolved through mediation shall be finally resolved by binding arbitration in accordance with the rules of [ARBITRATION ORGANIZATION]. The arbitration shall take place in [LOCATION].

4. Arbitration Procedures
   a) The arbitrator shall be selected by mutual agreement
   b) Discovery shall be limited to [SCOPE]
   c) The arbitrator's decision shall be final and binding
   d) Judgment may be entered in any court of competent jurisdiction

5. Exceptions
Notwithstanding the above, either party may seek injunctive relief from a court for breach of confidentiality, non-compete, or intellectual property provisions.

6. Costs
Each party shall bear its own legal fees unless the arbitrator determines otherwise."""},

        {"instruction": "Write a relocation assistance clause for an employment contract.",
         "output": """RELOCATION ASSISTANCE

1. Relocation Package
The Company shall provide Employee with relocation assistance for the move from [ORIGIN] to [DESTINATION] as follows:

2. Moving Expenses
The Company shall reimburse reasonable moving expenses up to $[AMOUNT], including:
   a) Professional moving services
   b) Packing materials and supplies
   c) Transportation of household goods
   d) Temporary storage up to [NUMBER] days

3. Travel Expenses
The Company shall cover:
   a) One house-hunting trip for Employee and spouse
   b) Final relocation travel for Employee and immediate family
   c) Temporary housing for up to [NUMBER] days

4. Additional Benefits
   a) Real estate commission assistance up to $[AMOUNT]
   b) Closing cost assistance up to $[AMOUNT]
   c) Lease break penalty reimbursement

5. Repayment Obligation
If Employee voluntarily terminates employment within [NUMBER] months of relocation, Employee shall repay relocation benefits on a pro-rata basis.

6. Tax Gross-Up
The Company shall provide tax gross-up for taxable relocation benefits."""},

        {"instruction": "Draft a training and professional development clause for employment.",
         "output": """TRAINING AND PROFESSIONAL DEVELOPMENT

1. Company-Sponsored Training
The Company may provide or require Employee to complete training programs, certifications, or courses related to their position.

2. Professional Development Budget
Employee shall be entitled to an annual professional development budget of $[AMOUNT] for:
   a) Industry conferences and seminars
   b) Professional certifications
   c) Online courses and educational materials
   d) Professional association memberships

3. Time Off for Training
Employee may request up to [NUMBER] days paid leave annually for approved training and professional development activities.

4. Training Agreement
For training exceeding $[AMOUNT] in cost, Employee shall sign a Training Agreement requiring:
   a) Completion of the training program
   b) Continued employment for [NUMBER] months following completion
   c) Repayment of costs if Employee leaves before the retention period

5. Knowledge Sharing
Employee agrees to share knowledge gained through training with relevant colleagues and teams.

6. Certification Maintenance
The Company shall support Employee in maintaining required professional certifications."""},

        {"instruction": "Write a performance review and evaluation clause for employment.",
         "output": """PERFORMANCE EVALUATION

1. Review Schedule
Employee shall receive formal performance reviews:
   a) Annually, typically in [MONTH]
   b) At the end of probationary period
   c) As needed for significant projects or concerns

2. Evaluation Criteria
Performance shall be evaluated based on:
   a) Achievement of established goals and objectives
   b) Quality of work and attention to detail
   c) Collaboration and teamwork
   d) Communication skills
   e) Initiative and problem-solving
   f) Adherence to company values and policies

3. Goal Setting
Employee and manager shall jointly establish measurable performance goals at the beginning of each review period.

4. Documentation
All performance reviews shall be documented in writing, reviewed with Employee, and signed by both parties.

5. Development Plans
When appropriate, performance reviews shall include individual development plans to address skill gaps or career growth.

6. Appeal Process
Employee may appeal performance ratings through HR within [NUMBER] days of receiving the review."""},
    ]
    dataset.extend(employment_templates)
    
    # -------------------------------------------------------------------------
    # 2. NON-DISCLOSURE AGREEMENTS (20 variations)
    # -------------------------------------------------------------------------
    nda_templates = [
        {"instruction": "Write the introduction and recitals for a mutual non-disclosure agreement between two technology companies.",
         "output": """MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement ("Agreement") is entered into as of [DATE] ("Effective Date"), by and between:

[COMPANY A NAME], a [STATE] corporation with its principal place of business at [ADDRESS] ("Company A"), and

[COMPANY B NAME], a [STATE] corporation with its principal place of business at [ADDRESS] ("Company B").

(Each a "Party" and collectively, the "Parties")

RECITALS

WHEREAS, Company A and Company B wish to explore a potential business relationship relating to [DESCRIPTION OF PURPOSE] (the "Purpose");

WHEREAS, in connection with the Purpose, each Party may disclose to the other certain confidential and proprietary information;

WHEREAS, the Parties wish to protect such confidential information from unauthorized use and disclosure;

NOW, THEREFORE, in consideration of the mutual promises and covenants contained herein, the Parties agree as follows:"""},
        
        {"instruction": "Draft the definition of confidential information for an NDA in the healthcare industry.",
         "output": """DEFINITION OF CONFIDENTIAL INFORMATION

1. "Confidential Information" means any and all information disclosed by either Party to the other, whether orally, in writing, electronically, or by any other means, including but not limited to:

   a) Technical Information: research data, clinical trial results, patient outcomes data, medical device specifications, pharmaceutical formulations, laboratory protocols, and scientific methodologies;

   b) Business Information: strategic plans, marketing strategies, pricing information, customer and patient lists, vendor relationships, and financial projections;

   c) Protected Health Information: any individually identifiable health information as defined under HIPAA and applicable state privacy laws;

   d) Intellectual Property: patents, patent applications, trade secrets, copyrights, inventions, discoveries, and proprietary processes;

   e) Regulatory Information: FDA submissions, regulatory correspondence, compliance documentation, and quality assurance records.

2. Confidential Information includes any summaries, analyses, compilations, or other materials prepared by the Receiving Party that contain or reflect Disclosing Party's Confidential Information."""},
        
        {"instruction": "Write the obligations section of a non-disclosure agreement.",
         "output": """OBLIGATIONS OF RECEIVING PARTY

1. Non-Disclosure
The Receiving Party shall:
   a) Hold and maintain Confidential Information in strict confidence
   b) Not disclose Confidential Information to any third party without prior written consent
   c) Use at least the same degree of care to protect Confidential Information as it uses for its own confidential information, but no less than reasonable care

2. Limited Use
The Receiving Party shall use Confidential Information solely for the Purpose and for no other purpose whatsoever.

3. Need-to-Know Basis
The Receiving Party shall limit disclosure of Confidential Information to its employees, contractors, and advisors who:
   a) Have a need to know such information for the Purpose
   b) Are bound by confidentiality obligations no less protective than those herein
   c) Have been informed of the confidential nature of such information

4. No Copies
The Receiving Party shall not copy or reproduce Confidential Information except as reasonably necessary for the Purpose.

5. Security Measures
The Receiving Party shall implement and maintain appropriate technical and organizational security measures to protect Confidential Information from unauthorized access, use, or disclosure."""},
        
        {"instruction": "Create exclusions from confidential information for an NDA.",
         "output": """EXCLUSIONS FROM CONFIDENTIAL INFORMATION

Confidential Information shall not include information that:

1. Public Domain
Is or becomes publicly available through no fault or action of the Receiving Party;

2. Prior Knowledge
Was rightfully in the Receiving Party's possession prior to disclosure by the Disclosing Party, as evidenced by written records;

3. Independent Development
Was independently developed by the Receiving Party without use of or reference to the Disclosing Party's Confidential Information, as evidenced by written records;

4. Third Party Disclosure
Is rightfully obtained by the Receiving Party from a third party without breach of any confidentiality obligation;

5. Written Authorization
Is disclosed with the prior written approval of the Disclosing Party.

BURDEN OF PROOF
The burden of proving that any of the above exclusions applies shall rest with the Party claiming the exclusion."""},
        
        {"instruction": "Draft the term and termination section of a non-disclosure agreement.",
         "output": """TERM AND TERMINATION

1. Term
This Agreement shall commence on the Effective Date and continue for a period of [NUMBER] years, unless earlier terminated as provided herein.

2. Termination for Convenience
Either Party may terminate this Agreement at any time upon [NUMBER] days' written notice to the other Party.

3. Termination for Breach
Either Party may terminate this Agreement immediately upon written notice if the other Party materially breaches this Agreement and fails to cure such breach within [NUMBER] days after receiving written notice.

4. Effect of Termination
Upon termination or expiration of this Agreement:
   a) Each Party shall promptly return or destroy all Confidential Information of the other Party
   b) Each Party shall certify in writing the return or destruction of such materials
   c) The obligations of confidentiality shall survive for [NUMBER] years following termination

5. Survival
The following provisions shall survive termination: Sections [LIST SECTIONS] (Confidentiality Obligations), [LIST SECTIONS] (Return of Materials), and [LIST SECTIONS] (Dispute Resolution)."""},
        
        {"instruction": "Write a required disclosure clause for an NDA covering legal and regulatory obligations.",
         "output": """REQUIRED DISCLOSURES

1. Legal Compulsion
If the Receiving Party is required by law, regulation, or legal process to disclose any Confidential Information, the Receiving Party shall:

   a) Provide prompt written notice to the Disclosing Party (to the extent legally permitted) to allow the Disclosing Party to seek a protective order or other remedy;

   b) Cooperate with the Disclosing Party, at the Disclosing Party's expense, in seeking such protective order;

   c) Disclose only the minimum amount of Confidential Information legally required;

   d) Use reasonable efforts to obtain confidential treatment of any disclosed Confidential Information.

2. Regulatory Requirements
Nothing in this Agreement shall prevent either Party from making disclosures required by:
   a) Securities laws and regulations
   b) Stock exchange rules
   c) Government regulatory agencies
   d) Court orders or subpoenas

3. Prior Notice
Except where prohibited by law, the Receiving Party shall provide at least [NUMBER] business days' prior written notice before making any required disclosure."""},

        {"instruction": "Draft intellectual property rights provisions for an NDA.",
         "output": """INTELLECTUAL PROPERTY RIGHTS

1. No License Granted
Nothing in this Agreement shall be construed as granting any license, either express or implied, to any Confidential Information or any patent, trademark, copyright, trade secret, or other intellectual property right.

2. Ownership
All Confidential Information shall remain the sole property of the Disclosing Party. The Receiving Party acquires no rights in such Confidential Information except the limited right to use it for the Purpose.

3. No Obligation
This Agreement does not obligate either Party to:
   a) Enter into any further agreement
   b) Proceed with any transaction or business relationship
   c) Disclose any particular Confidential Information

4. Derivative Works
Any analyses, compilations, or other materials created by the Receiving Party containing Confidential Information:
   a) Shall be deemed Confidential Information of the Disclosing Party
   b) Shall be subject to all protections of this Agreement
   c) Shall be returned or destroyed upon request

5. Residual Knowledge
Nothing herein shall restrict the Receiving Party's use of general knowledge, skills, and experience retained in the unaided memory of its personnel."""},

        {"instruction": "Write the remedies and enforcement section of an NDA.",
         "output": """REMEDIES AND ENFORCEMENT

1. Irreparable Harm
The Parties acknowledge that unauthorized disclosure or use of Confidential Information may cause irreparable harm for which monetary damages may be inadequate.

2. Injunctive Relief
In the event of any breach or threatened breach, the Disclosing Party shall be entitled to seek:
   a) Temporary restraining orders
   b) Preliminary and permanent injunctions
   c) Specific performance
   
Without the necessity of proving actual damages or posting a bond.

3. Monetary Damages
The Disclosing Party shall also be entitled to recover:
   a) Actual damages suffered as a result of the breach
   b) Consequential damages to the extent foreseeable
   c) Disgorgement of profits derived from unauthorized use

4. Cumulative Remedies
All remedies under this Agreement are cumulative and not exclusive of any other remedies available at law or in equity.

5. Prevailing Party
In any action to enforce this Agreement, the prevailing Party shall be entitled to recover reasonable attorneys' fees and costs."""},

        {"instruction": "Create a unilateral non-disclosure agreement for a company sharing information with a contractor.",
         "output": """UNILATERAL NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("Agreement") is made as of [DATE] between:

[COMPANY NAME] ("Disclosing Party"), and
[CONTRACTOR NAME] ("Receiving Party").

1. PURPOSE
The Disclosing Party wishes to disclose certain confidential information to the Receiving Party for the purpose of [DESCRIPTION OF PROJECT/ENGAGEMENT].

2. CONFIDENTIAL INFORMATION
"Confidential Information" includes all information disclosed by the Disclosing Party, including but not limited to: technical data, trade secrets, business plans, customer information, and proprietary methodologies.

3. OBLIGATIONS
The Receiving Party agrees to:
   a) Maintain all Confidential Information in strict confidence
   b) Use Confidential Information solely for the stated Purpose
   c) Not disclose to third parties without prior written consent
   d) Protect Confidential Information with reasonable security measures

4. TERM
This Agreement shall remain in effect for [NUMBER] years from the Effective Date. Confidentiality obligations shall survive for [NUMBER] years after termination.

5. RETURN OF MATERIALS
Upon request or termination, the Receiving Party shall return or destroy all Confidential Information and certify such action in writing."""},

        {"instruction": "Draft an NDA clause for protecting software source code.",
         "output": """SOURCE CODE PROTECTION

1. Source Code Definition
"Source Code" means the human-readable form of computer programming code, including all related documentation, comments, specifications, algorithms, and development tools.

2. Highest Level of Protection
Source Code shall be afforded the highest level of protection under this Agreement and shall be subject to the following additional requirements:

3. Access Restrictions
   a) Access to Source Code shall be limited to named individuals approved in writing
   b) Source Code shall be accessed only on secure, Company-approved systems
   c) No copies of Source Code shall be made without express written authorization

4. Technical Safeguards
The Receiving Party shall implement:
   a) Encrypted storage for all Source Code
   b) Access logging and monitoring
   c) Secure development environments
   d) Code repository access controls

5. Prohibited Actions
The Receiving Party shall not:
   a) Reverse engineer, decompile, or disassemble any compiled code
   b) Create derivative works from the Source Code
   c) Use Source Code to develop competing products
   d) Share or discuss Source Code in public forums

6. Audit Rights
The Disclosing Party reserves the right to audit compliance with Source Code protection requirements upon reasonable notice."""},
    ]
    dataset.extend(nda_templates)
    
    # -------------------------------------------------------------------------
    # 3. PRIVACY POLICIES (20 variations)
    # -------------------------------------------------------------------------
    privacy_templates = [
        {"instruction": "Write the introduction section of a privacy policy for a mobile application.",
         "output": """PRIVACY POLICY

Last Updated: [DATE]

Welcome to [APP NAME]. This Privacy Policy describes how [COMPANY NAME] ("we," "us," or "our") collects, uses, shares, and protects information about users ("you" or "your") of our mobile application.

We are committed to protecting your privacy and handling your data in an open and transparent manner. Please read this Privacy Policy carefully to understand our practices regarding your personal information.

By downloading, installing, or using our application, you acknowledge that you have read and understood this Privacy Policy. If you do not agree with our practices, please do not use our application.

This Privacy Policy applies to information we collect:
- Through our mobile application
- In email, text, and other electronic communications
- Through interactions with our customer support
- When you interact with our advertising and applications on third-party websites and services"""},
        
        {"instruction": "Draft the data collection section of a privacy policy for an e-commerce website.",
         "output": """INFORMATION WE COLLECT

We collect several types of information from and about users of our website:

1. Information You Provide Directly
   - Account information: name, email address, username, password
   - Purchase information: billing address, shipping address, payment details
   - Profile information: preferences, wishlist items, product reviews
   - Communications: customer service inquiries, feedback, survey responses

2. Information Collected Automatically
   - Device information: IP address, browser type, operating system, device identifiers
   - Usage data: pages visited, time spent, click patterns, search queries
   - Transaction data: products viewed, items added to cart, purchase history
   - Location data: general location based on IP address

3. Information from Third Parties
   - Social media: information from connected social accounts
   - Marketing partners: demographic and interest data
   - Payment processors: transaction verification data
   - Fraud prevention services: risk assessment data

4. Cookies and Tracking Technologies
We use cookies, web beacons, and similar technologies to collect information about your browsing activities. See our Cookie Policy for details."""},
        
        {"instruction": "Write the data usage section of a privacy policy for a SaaS platform.",
         "output": """HOW WE USE YOUR INFORMATION

We use the information we collect for the following purposes:

1. Service Delivery
   - Providing, maintaining, and improving our platform
   - Processing transactions and sending related information
   - Managing your account and providing customer support
   - Personalizing your experience and content

2. Communication
   - Sending service-related notices and updates
   - Responding to your comments, questions, and requests
   - Providing customer service and technical support
   - Sending promotional communications (with your consent)

3. Analytics and Improvement
   - Analyzing usage patterns to improve our services
   - Conducting research and development
   - Monitoring and analyzing trends and user behavior
   - Testing new features and functionality

4. Security and Compliance
   - Detecting, preventing, and addressing fraud and abuse
   - Protecting the rights and safety of users and third parties
   - Complying with legal obligations and enforcing our terms
   - Responding to law enforcement requests and legal processes

5. Advertising and Marketing
   - Delivering targeted advertising based on your interests
   - Measuring advertising effectiveness
   - Conducting marketing research and analysis"""},
        
        {"instruction": "Create the data sharing section of a privacy policy explaining third-party disclosures.",
         "output": """HOW WE SHARE YOUR INFORMATION

We may share your personal information in the following circumstances:

1. Service Providers
We share information with third-party vendors who provide services on our behalf, including:
   - Cloud hosting and data storage
   - Payment processing
   - Email delivery
   - Analytics and advertising
   - Customer support tools

These providers are contractually obligated to use your information only for the services they provide to us.

2. Business Transfers
In the event of a merger, acquisition, reorganization, or sale of assets, your information may be transferred as part of that transaction.

3. Legal Requirements
We may disclose information when required by law or in response to:
   - Court orders and subpoenas
   - Government requests
   - To protect our legal rights
   - In emergencies involving potential threats to safety

4. With Your Consent
We may share your information with third parties when you have given us explicit consent to do so.

5. Aggregated or De-identified Data
We may share aggregated or de-identified information that cannot reasonably be used to identify you.

WE DO NOT SELL YOUR PERSONAL INFORMATION."""},
        
        {"instruction": "Write the data retention section of a privacy policy.",
         "output": """DATA RETENTION

1. Retention Periods
We retain your personal information for as long as necessary to fulfill the purposes outlined in this Privacy Policy, unless a longer retention period is required or permitted by law.

   - Account information: Retained while your account is active and for [NUMBER] years after closure
   - Transaction records: Retained for [NUMBER] years for tax and legal compliance
   - Communication records: Retained for [NUMBER] years
   - Usage data: Retained for [NUMBER] months for analytics purposes

2. Retention Criteria
We consider the following factors when determining retention periods:
   - The nature and sensitivity of the data
   - The purposes for which we process the data
   - Legal, regulatory, and contractual requirements
   - Our legitimate business interests

3. Deletion
When personal information is no longer needed, we will:
   - Securely delete or destroy the information, or
   - Anonymize the information so it can no longer identify you

4. Backup Systems
Information in backup systems may be retained for additional periods for disaster recovery purposes but will be deleted in accordance with our backup rotation schedule.

5. Legal Holds
We may retain information longer if required for legal proceedings or investigations."""},
        
        {"instruction": "Draft the user rights section of a GDPR-compliant privacy policy.",
         "output": """YOUR RIGHTS AND CHOICES

Under applicable data protection laws, you have the following rights:

1. Right of Access
You have the right to request a copy of the personal information we hold about you.

2. Right to Rectification
You have the right to request correction of inaccurate or incomplete personal information.

3. Right to Erasure ("Right to be Forgotten")
You have the right to request deletion of your personal information in certain circumstances, such as when the data is no longer necessary for its original purpose.

4. Right to Restrict Processing
You have the right to request that we limit how we use your personal information.

5. Right to Data Portability
You have the right to receive your personal information in a structured, commonly used, machine-readable format.

6. Right to Object
You have the right to object to processing of your personal information for direct marketing purposes or based on legitimate interests.

7. Rights Related to Automated Decision-Making
You have the right not to be subject to decisions based solely on automated processing that significantly affect you.

HOW TO EXERCISE YOUR RIGHTS
To exercise any of these rights, please contact us at [EMAIL]. We will respond within [30] days.

You also have the right to lodge a complaint with your local data protection authority."""},

        {"instruction": "Write the security measures section of a privacy policy.",
         "output": """DATA SECURITY

We implement appropriate technical and organizational measures to protect your personal information against unauthorized access, alteration, disclosure, or destruction.

1. Technical Safeguards
   - Encryption of data in transit using TLS/SSL
   - Encryption of sensitive data at rest using AES-256
   - Regular security assessments and penetration testing
   - Intrusion detection and prevention systems
   - Secure access controls and authentication

2. Organizational Measures
   - Employee training on data protection and security
   - Background checks for personnel with data access
   - Confidentiality agreements with staff and contractors
   - Incident response procedures
   - Regular policy reviews and updates

3. Access Controls
   - Role-based access restrictions
   - Multi-factor authentication for administrative access
   - Regular access reviews and audits
   - Principle of least privilege

4. Physical Security
   - Secure data center facilities
   - Access controls and monitoring
   - Environmental controls and redundancy

IMPORTANT NOTICE
While we strive to protect your personal information, no method of transmission over the Internet or electronic storage is 100% secure. We cannot guarantee absolute security."""},

        {"instruction": "Create a children's privacy section for a family-oriented app.",
         "output": """CHILDREN'S PRIVACY

Our application is designed for users of all ages, including children under 13. We are committed to complying with the Children's Online Privacy Protection Act (COPPA) and other applicable laws.

1. Parental Consent
For users under 13, we require verifiable parental consent before collecting any personal information. Parents may provide consent through:
   - Credit card verification
   - Signed consent form
   - Video conference verification

2. Information We Collect from Children
With parental consent, we may collect:
   - Username (not real name)
   - Parent's email address for notifications
   - Age or age range
   - Usage data for service improvement

3. Information We Do Not Collect
We do not knowingly collect from children:
   - Real names or addresses
   - Photographs or videos
   - Precise geolocation data
   - Persistent identifiers for advertising

4. Parental Rights
Parents or guardians have the right to:
   - Review their child's personal information
   - Request deletion of their child's data
   - Refuse further collection or use
   - Revoke consent at any time

5. Contact Us
For questions about children's privacy, contact us at [CHILDREN'S PRIVACY EMAIL].

If we learn that we have collected personal information from a child under 13 without parental consent, we will delete that information immediately."""},

        {"instruction": "Write the cookie policy section of a website privacy policy.",
         "output": """COOKIES AND TRACKING TECHNOLOGIES

1. What Are Cookies?
Cookies are small text files placed on your device when you visit our website. They help us provide you with a better experience and allow certain features to work.

2. Types of Cookies We Use

   Essential Cookies
   Required for basic site functionality. Cannot be disabled.
   - Session management
   - Security features
   - Shopping cart functionality

   Performance Cookies
   Help us understand how visitors use our site.
   - Google Analytics
   - Error tracking
   - Load balancing

   Functionality Cookies
   Remember your preferences and settings.
   - Language preferences
   - Display settings
   - Login information

   Advertising Cookies
   Used to deliver relevant advertisements.
   - Interest-based advertising
   - Retargeting
   - Ad measurement

3. Managing Cookies
You can control cookies through:
   - Browser settings (blocking or deleting cookies)
   - Our cookie consent tool
   - Opt-out links for specific providers

4. Do Not Track
We currently do not respond to Do Not Track browser signals. We will update this policy if we implement Do Not Track response in the future.

5. Third-Party Cookies
We use services from third parties that may set their own cookies. Please review their privacy policies for more information."""},

        {"instruction": "Draft the international data transfer section of a privacy policy.",
         "output": """INTERNATIONAL DATA TRANSFERS

1. Location of Processing
We are headquartered in [COUNTRY]. Your personal information may be transferred to, stored, and processed in countries other than your country of residence, including the United States and other countries where we or our service providers operate.

2. Legal Basis for Transfers
When we transfer personal information internationally, we rely on:
   - Standard Contractual Clauses approved by the European Commission
   - Adequacy decisions for countries deemed to provide adequate protection
   - Your explicit consent, where applicable
   - Necessary transfers for contract performance

3. Standard Contractual Clauses
We have implemented Standard Contractual Clauses with our service providers to ensure appropriate safeguards for your personal information.

4. Privacy Shield
While Privacy Shield is no longer valid for EU-US transfers, we continue to adhere to Privacy Shield Principles where applicable.

5. Your Rights
Regardless of where your information is processed, we will protect it in accordance with this Privacy Policy and applicable laws.

6. Additional Safeguards
We implement supplementary measures as needed, including:
   - Data encryption
   - Pseudonymization
   - Enhanced security controls
   - Transfer impact assessments"""},
    ]
    dataset.extend(privacy_templates)
    
    # -------------------------------------------------------------------------
    # 4. BUSINESS EMAILS (25 variations)
    # -------------------------------------------------------------------------
    email_templates = [
        {"instruction": "Write a professional email introducing a company's services to a potential client.",
         "output": """Subject: Innovative Solutions for Your Business Growth

Dear [RECIPIENT NAME],

I hope this email finds you well. My name is [YOUR NAME], and I am the [TITLE] at [COMPANY NAME], a leading provider of [INDUSTRY] solutions.

I recently came across [PROSPECT COMPANY] and was impressed by your work in [SPECIFIC AREA]. I believe there may be an excellent opportunity for us to collaborate.

At [COMPANY NAME], we specialize in:
• [SERVICE 1] - helping businesses achieve [BENEFIT]
• [SERVICE 2] - enabling organizations to [BENEFIT]
• [SERVICE 3] - providing [BENEFIT]

Our clients, including [NOTABLE CLIENT NAMES], have achieved [SPECIFIC RESULTS] through our partnership.

I would welcome the opportunity to schedule a brief call to discuss how we might support [PROSPECT COMPANY]'s goals. Would you have 20 minutes available next week for an introductory conversation?

Please feel free to reach me directly at [PHONE] or reply to this email.

Thank you for your time, and I look forward to the possibility of working together.

Best regards,

[YOUR NAME]
[TITLE]
[COMPANY NAME]
[PHONE] | [EMAIL]"""},
        
        {"instruction": "Draft a professional email requesting a deadline extension from a client.",
         "output": """Subject: Request for Project Deadline Extension - [PROJECT NAME]

Dear [CLIENT NAME],

I hope this message finds you well. I am writing to discuss the timeline for [PROJECT NAME].

After careful review of our progress and remaining deliverables, I would like to request an extension of [NUMBER] days to our original deadline of [ORIGINAL DATE]. This would move our delivery date to [NEW DATE].

The reasons for this request are:
1. [REASON 1 - e.g., additional scope identified during development]
2. [REASON 2 - e.g., technical complexity exceeded initial estimates]
3. [REASON 3 - e.g., dependency on third-party integration]

I want to assure you that our team remains fully committed to delivering a high-quality product. This extension will allow us to:
• Thoroughly test all functionality
• Address edge cases we've identified
• Ensure the solution meets all specifications

To minimize the impact, we can:
• Provide a progress demo by [DATE]
• Deliver Phase 1 features by the original deadline
• Provide daily status updates during the extension period

I understand this may affect your planning, and I sincerely apologize for any inconvenience. Please let me know if you would like to discuss this further or if you have any concerns.

Thank you for your understanding and continued partnership.

Best regards,

[YOUR NAME]"""},
        
        {"instruction": "Write a follow-up email after a business meeting.",
         "output": """Subject: Follow-Up: [MEETING TOPIC] Discussion - Next Steps

Dear [RECIPIENT NAME],

Thank you for taking the time to meet with me [today/yesterday] to discuss [TOPIC]. I thoroughly enjoyed our conversation and learning more about [THEIR COMPANY/PROJECT].

As promised, I wanted to summarize the key points we discussed and outline the next steps:

Key Discussion Points:
• [POINT 1]
• [POINT 2]
• [POINT 3]

Agreed Action Items:
1. [ACTION] - Owner: [NAME] - Due: [DATE]
2. [ACTION] - Owner: [NAME] - Due: [DATE]
3. [ACTION] - Owner: [NAME] - Due: [DATE]

I have attached [DOCUMENT/PROPOSAL/INFORMATION] that we discussed during our meeting for your reference.

Our next meeting is scheduled for [DATE] at [TIME]. Please let me know if you need to reschedule or if there are additional topics you would like to add to the agenda.

If you have any questions in the meantime, please don't hesitate to reach out.

Thank you again for your time, and I look forward to our continued collaboration.

Best regards,

[YOUR NAME]"""},
        
        {"instruction": "Create an email apologizing for a service disruption to customers.",
         "output": """Subject: Our Apology - Service Disruption on [DATE]

Dear Valued Customer,

I am writing to personally apologize for the service disruption you experienced on [DATE] between [TIME] and [TIME].

What Happened:
Our platform experienced [BRIEF TECHNICAL EXPLANATION] which resulted in [IMPACT - e.g., inability to access accounts, delayed transactions].

What We Did:
Our engineering team was alerted immediately and worked around the clock to:
• Identify the root cause within [TIME]
• Implement a fix by [TIME]
• Restore full service by [TIME]

How We're Preventing This:
We take this incident seriously. To prevent similar occurrences, we are:
1. [PREVENTIVE MEASURE 1]
2. [PREVENTIVE MEASURE 2]
3. [PREVENTIVE MEASURE 3]

Making It Right:
As a token of our appreciation for your patience and understanding, we are [COMPENSATION - e.g., crediting your account, extending your subscription, providing a discount].

We understand that you depend on our service, and we deeply regret any inconvenience this may have caused. Your trust means everything to us.

If you have any questions or concerns, please contact our support team at [CONTACT].

Sincerely,

[NAME]
[TITLE]
[COMPANY]"""},
        
        {"instruction": "Draft an email declining a business proposal politely.",
         "output": """Subject: Re: Partnership Proposal - [PROPOSAL NAME]

Dear [NAME],

Thank you for taking the time to present your proposal for [PROPOSAL DESCRIPTION]. We appreciate your interest in partnering with [OUR COMPANY] and the effort you put into preparing this presentation.

After careful consideration by our team, we have decided not to move forward with this opportunity at this time.

This decision was not made lightly. While we recognize the potential value of [SPECIFIC ASPECT OF PROPOSAL], we found that:
• [REASON 1 - e.g., the timing does not align with our current strategic priorities]
• [REASON 2 - e.g., resource constraints prevent us from committing fully]

Please understand that this decision reflects our current circumstances and does not diminish our respect for your organization and the quality of your proposal.

We would be open to:
• Revisiting this conversation in [TIMEFRAME]
• Exploring alternative collaboration opportunities
• Staying connected for potential future projects

Thank you again for considering [OUR COMPANY]. We wish you every success with this initiative and hope our paths may cross again in the future.

Please feel free to stay in touch.

Best regards,

[YOUR NAME]
[TITLE]"""},
        
        {"instruction": "Write a professional email announcing a price increase to customers.",
         "output": """Subject: Important Update to Our Pricing - Effective [DATE]

Dear [CUSTOMER NAME],

I am writing to inform you of an upcoming change to our pricing structure, effective [DATE].

What's Changing:
Our [PRODUCT/SERVICE] pricing will increase by [PERCENTAGE/AMOUNT]:
• [CURRENT PLAN]: from $[OLD PRICE] to $[NEW PRICE]
• [CURRENT PLAN]: from $[OLD PRICE] to $[NEW PRICE]

Why We're Making This Change:
Over the past [TIME PERIOD], we have made significant investments in:
• [IMPROVEMENT 1 - e.g., enhanced security features]
• [IMPROVEMENT 2 - e.g., 24/7 customer support]
• [IMPROVEMENT 3 - e.g., new functionality and integrations]

These improvements reflect our commitment to providing you with the best possible service.

What This Means for You:
• Your current rate will remain in effect until [DATE]
• No action is required on your part
• All features you currently enjoy will continue

Our Commitment:
We remain dedicated to delivering exceptional value. If you have any questions or would like to discuss your account, please contact your account manager at [CONTACT].

Thank you for your continued trust in [COMPANY NAME]. We greatly value your partnership and look forward to serving you.

Sincerely,

[NAME]
[TITLE]"""},

        {"instruction": "Create an email requesting payment for an overdue invoice.",
         "output": """Subject: Reminder: Invoice #[NUMBER] - Payment Overdue

Dear [CLIENT NAME],

I hope this message finds you well. I am writing to follow up on Invoice #[NUMBER] dated [DATE] in the amount of $[AMOUNT], which was due on [DUE DATE].

As of today, this invoice is [NUMBER] days overdue, and we have not yet received payment.

Invoice Details:
• Invoice Number: [NUMBER]
• Invoice Date: [DATE]
• Amount Due: $[AMOUNT]
• Original Due Date: [DATE]
• Days Overdue: [NUMBER]

For your convenience, I have attached a copy of the invoice to this email. Payment can be made via:
• Bank transfer to [ACCOUNT DETAILS]
• Check payable to [COMPANY NAME]
• Online at [PAYMENT PORTAL]

If you have already sent payment, please disregard this reminder, and thank you. If there are any issues with the invoice or if you need to discuss payment arrangements, please contact me at your earliest convenience.

We value our business relationship and would like to resolve this matter promptly.

Thank you for your attention to this matter.

Best regards,

[YOUR NAME]
[TITLE]
[COMPANY NAME]
[PHONE] | [EMAIL]"""},

        {"instruction": "Write an email announcing a new product launch to existing customers.",
         "output": """Subject: Introducing [PRODUCT NAME] - Exclusively for Our Valued Customers

Dear [CUSTOMER NAME],

We are thrilled to announce the launch of [PRODUCT NAME] - our most [INNOVATIVE/POWERFUL/ADVANCED] solution yet!

After [TIME PERIOD] of development and listening to feedback from customers like you, we have created a [PRODUCT TYPE] that [KEY VALUE PROPOSITION].

What Makes [PRODUCT NAME] Special:

✨ [FEATURE 1]: [Brief description of benefit]
✨ [FEATURE 2]: [Brief description of benefit]
✨ [FEATURE 3]: [Brief description of benefit]
✨ [FEATURE 4]: [Brief description of benefit]

Exclusive Early Access Offer:
As a valued customer, you have exclusive early access to [PRODUCT NAME]. For the next [TIME PERIOD], enjoy:
• [OFFER 1 - e.g., 20% launch discount]
• [OFFER 2 - e.g., free premium support]
• [OFFER 3 - e.g., complimentary training]

Use code [PROMO CODE] at checkout.

[CTA BUTTON: Try [PRODUCT NAME] Now]

Want to Learn More?
• Watch our product demo: [LINK]
• Read the full announcement: [LINK]
• Join our launch webinar on [DATE]: [LINK]

Thank you for being part of our journey. We can't wait for you to experience [PRODUCT NAME].

Questions? Reply to this email or contact [SUPPORT CONTACT].

Best regards,

[NAME]
[TITLE]
[COMPANY]"""},

        {"instruction": "Draft an email requesting a meeting with a senior executive.",
         "output": """Subject: Meeting Request: [TOPIC] Discussion

Dear [EXECUTIVE NAME],

I hope this email finds you well. My name is [YOUR NAME], [YOUR TITLE] at [YOUR COMPANY/DEPARTMENT].

I am reaching out to request a brief meeting to discuss [TOPIC/INITIATIVE]. Given your expertise in [RELEVANT AREA], I believe your insights would be invaluable.

Purpose of Meeting:
[BRIEF DESCRIPTION - 2-3 sentences explaining the context and why this meeting is important]

Specifically, I would like to discuss:
1. [AGENDA ITEM 1]
2. [AGENDA ITEM 2]
3. [AGENDA ITEM 3]

I anticipate the meeting would take approximately [30 minutes/1 hour].

I am flexible with timing and happy to work around your schedule. Some times that work for me include:
• [DATE/TIME OPTION 1]
• [DATE/TIME OPTION 2]
• [DATE/TIME OPTION 3]

Alternatively, please let me know what works best for you, and I will adjust accordingly.

I have attached [RELEVANT DOCUMENT] for your review prior to our meeting.

Thank you for considering this request. I understand your time is valuable and greatly appreciate any opportunity to connect.

Best regards,

[YOUR NAME]
[TITLE]
[COMPANY/DEPARTMENT]
[PHONE] | [EMAIL]"""},

        {"instruction": "Write a professional email providing a project status update.",
         "output": """Subject: Weekly Project Update: [PROJECT NAME] - Week of [DATE]

Dear [STAKEHOLDER NAME],

Please find below the weekly status update for [PROJECT NAME].

OVERALL STATUS: [🟢 On Track / 🟡 At Risk / 🔴 Off Track]

EXECUTIVE SUMMARY
[2-3 sentences summarizing the overall project health and key highlights]

PROGRESS THIS WEEK
✅ Completed:
• [MILESTONE/TASK 1]
• [MILESTONE/TASK 2]
• [MILESTONE/TASK 3]

🔄 In Progress:
• [TASK 1] - [PERCENTAGE]% complete
• [TASK 2] - [PERCENTAGE]% complete

UPCOMING MILESTONES
• [MILESTONE 1] - Due: [DATE]
• [MILESTONE 2] - Due: [DATE]

RISKS AND ISSUES
| Risk/Issue | Impact | Mitigation | Owner |
|------------|--------|------------|-------|
| [DESCRIPTION] | [HIGH/MED/LOW] | [ACTION] | [NAME] |

BUDGET STATUS
• Allocated: $[AMOUNT]
• Spent to Date: $[AMOUNT]
• Remaining: $[AMOUNT]

DECISIONS NEEDED
1. [DECISION REQUIRED] - Deadline: [DATE]

Please let me know if you have any questions or require additional details.

Best regards,

[YOUR NAME]
Project Manager"""},
    ]
    dataset.extend(email_templates)
    
    # -------------------------------------------------------------------------
    # 5. MEETING MINUTES (15 variations)
    # -------------------------------------------------------------------------
    minutes_templates = [
        {"instruction": "Write meeting minutes for a weekly project status meeting.",
         "output": """MEETING MINUTES

Project: [PROJECT NAME]
Meeting Type: Weekly Status Meeting
Date: [DATE]
Time: [START TIME] - [END TIME]
Location: [LOCATION/VIDEO CONFERENCE]

ATTENDEES
Present: [NAME 1] (Chair), [NAME 2], [NAME 3], [NAME 4]
Absent: [NAME 5] (excused)

AGENDA
1. Review of previous action items
2. Progress updates by workstream
3. Risk and issue discussion
4. Upcoming milestones
5. Open discussion

MEETING SUMMARY

1. Review of Previous Action Items
   • [ACTION ITEM 1]: Completed by [NAME]
   • [ACTION ITEM 2]: In progress, expected completion [DATE]
   • [ACTION ITEM 3]: Carried over to next week

2. Progress Updates
   Development Team ([NAME]):
   - Completed [FEATURE/TASK]
   - Currently working on [FEATURE/TASK]
   - On track for [MILESTONE] deadline

   Design Team ([NAME]):
   - Finalized [DELIVERABLE]
   - Awaiting feedback on [ITEM]

3. Risks and Issues
   - [RISK]: [DESCRIPTION]. Mitigation: [ACTION]
   - [ISSUE]: [DESCRIPTION]. Resolution: [ACTION] by [DATE]

4. Upcoming Milestones
   - [DATE]: [MILESTONE DESCRIPTION]
   - [DATE]: [MILESTONE DESCRIPTION]

NEW ACTION ITEMS
| # | Action | Owner | Due Date |
|---|--------|-------|----------|
| 1 | [DESCRIPTION] | [NAME] | [DATE] |
| 2 | [DESCRIPTION] | [NAME] | [DATE] |

NEXT MEETING
Date: [DATE], Time: [TIME], Location: [LOCATION]

Minutes prepared by: [NAME]
Date: [DATE]"""},
        
        {"instruction": "Create meeting minutes for a board of directors meeting.",
         "output": """MINUTES OF THE MEETING OF THE BOARD OF DIRECTORS
[COMPANY NAME]

A [regular/special] meeting of the Board of Directors was held on [DATE] at [TIME] at [LOCATION/via video conference].

DIRECTORS PRESENT
[NAME], Chairman
[NAME], Director
[NAME], Director
[NAME], Director

ALSO PRESENT
[NAME], Chief Executive Officer
[NAME], Chief Financial Officer
[NAME], Corporate Secretary

QUORUM
The Chairman confirmed that a quorum was present and called the meeting to order at [TIME].

APPROVAL OF PREVIOUS MINUTES
Upon motion duly made by [NAME] and seconded by [NAME], the minutes of the meeting held on [DATE] were unanimously approved as presented.

FINANCIAL REPORT
The CFO presented the financial statements for [PERIOD], reporting:
• Revenue: $[AMOUNT] ([PERCENTAGE]% [above/below] budget)
• Net Income: $[AMOUNT]
• Cash Position: $[AMOUNT]

Discussion ensued regarding [TOPIC]. The Board [approved/noted] the financial report.

RESOLUTION: Upon motion by [NAME], seconded by [NAME], it was RESOLVED that the financial statements be approved as presented. Motion carried unanimously.

EXECUTIVE SESSION
At [TIME], the Board entered executive session. All non-Board members except [NAME] were excused.

[SUMMARY OF EXECUTIVE SESSION MATTERS]

The executive session concluded at [TIME].

ADJOURNMENT
There being no further business, upon motion duly made and seconded, the meeting was adjourned at [TIME].

_______________________
[NAME], Corporate Secretary

Approved: _______________
Date: _______________"""},
        
        {"instruction": "Draft meeting minutes for a client kickoff meeting.",
         "output": """CLIENT KICKOFF MEETING MINUTES

Project: [PROJECT NAME]
Client: [CLIENT COMPANY]
Date: [DATE]
Location: [LOCATION]

ATTENDEES
Client Representatives:
• [NAME], [TITLE]
• [NAME], [TITLE]

[YOUR COMPANY] Team:
• [NAME], Project Manager
• [NAME], Technical Lead
• [NAME], Account Manager

PURPOSE
Official kickoff meeting to align on project objectives, timeline, and governance.

DISCUSSION SUMMARY

1. Project Overview and Objectives
   [NAME] presented the project scope:
   • Primary objective: [DESCRIPTION]
   • Key deliverables: [LIST]
   • Success criteria: [METRICS]
   
   Client confirmed alignment with stated objectives.

2. Timeline and Milestones
   | Phase | Deliverable | Target Date |
   |-------|-------------|-------------|
   | Discovery | Requirements document | [DATE] |
   | Design | Approved mockups | [DATE] |
   | Development | Beta release | [DATE] |
   | Launch | Go-live | [DATE] |

3. Team Structure and Governance
   • Weekly status meetings: [DAY] at [TIME]
   • Primary contacts: [NAMES]
   • Escalation path: [PROCESS]
   • Communication tools: [EMAIL/SLACK/ETC]

4. Client Responsibilities
   • Provide access to [SYSTEMS/INFORMATION]
   • Designate subject matter experts
   • Review and approve deliverables within [X] business days

5. Risks and Assumptions
   Discussed and documented in project charter.

ACTION ITEMS
| Action | Owner | Due |
|--------|-------|-----|
| Distribute project charter | [NAME] | [DATE] |
| Schedule recurring meetings | [NAME] | [DATE] |
| Provide system access | [CLIENT NAME] | [DATE] |

NEXT STEPS
• Requirements gathering sessions begin [DATE]
• First status meeting: [DATE]

Minutes recorded by: [NAME]"""},
    ]
    dataset.extend(minutes_templates)
    
    # -------------------------------------------------------------------------
    # 6. RECOMMENDATION LETTERS (15 variations)
    # -------------------------------------------------------------------------
    recommendation_templates = [
        {"instruction": "Write a recommendation letter for a software engineer applying to graduate school.",
         "output": """[YOUR LETTERHEAD/CONTACT INFO]

[DATE]

Graduate Admissions Committee
[UNIVERSITY NAME]
[ADDRESS]

Dear Members of the Admissions Committee,

I am writing to enthusiastically recommend [CANDIDATE NAME] for admission to your [PROGRAM NAME] program. As [YOUR TITLE] at [COMPANY], I have had the pleasure of supervising [CANDIDATE] for [DURATION] and can speak directly to their exceptional abilities and potential for graduate study.

[CANDIDATE] joined our team as a Software Engineer and quickly distinguished themselves through their technical excellence and intellectual curiosity. Their work on [SPECIFIC PROJECT] demonstrated sophisticated understanding of [TECHNICAL AREA] and resulted in [QUANTIFIABLE OUTCOME].

What sets [CANDIDATE] apart is their research mindset. They approach problems systematically, designing experiments to test hypotheses and rigorously analyzing results. When faced with a challenge in [SPECIFIC AREA], [CANDIDATE] independently studied relevant academic literature and implemented a novel solution that [OUTCOME].

Key strengths include:
• Strong foundation in [TECHNICAL AREAS]
• Excellent written and verbal communication
• Ability to translate complex concepts for diverse audiences
• Self-motivated learner who actively seeks growth opportunities

[CANDIDATE] has consistently sought opportunities to expand their knowledge beyond day-to-day responsibilities. They have [EXAMPLES: completed online courses, attended conferences, contributed to open-source projects].

I have no doubt that [CANDIDATE] possesses the intellectual capability, work ethic, and genuine passion for research needed to excel in graduate study. They would be an asset to your program and the broader academic community.

Please do not hesitate to contact me if you require any additional information.

Sincerely,

[YOUR NAME]
[YOUR TITLE]
[COMPANY]
[EMAIL] | [PHONE]"""},
        
        {"instruction": "Draft a professional recommendation letter for a marketing manager seeking a new position.",
         "output": """[LETTERHEAD]

[DATE]

To Whom It May Concern,

It is with great pleasure that I recommend [CANDIDATE NAME] for any marketing leadership position. As [YOUR TITLE] at [COMPANY], I worked closely with [CANDIDATE] for [DURATION] and observed firsthand their exceptional marketing expertise and leadership capabilities.

[CANDIDATE] served as Marketing Manager, where they were responsible for [SCOPE OF RESPONSIBILITIES]. Under their leadership, our marketing team achieved:

• [METRIC]% increase in brand awareness
• $[AMOUNT] in marketing-attributed revenue
• [PERCENTAGE]% improvement in lead generation
• Successful launch of [NUMBER] major campaigns

One particularly impressive achievement was [SPECIFIC PROJECT/CAMPAIGN]. [CANDIDATE] identified [OPPORTUNITY/CHALLENGE] and developed a comprehensive strategy that [OUTCOME]. This initiative demonstrated their ability to think strategically while executing flawlessly.

Beyond technical marketing skills, [CANDIDATE] excels as a leader. They built a high-performing team, mentoring junior staff and fostering a collaborative environment. Team members consistently praised their supportive management style and clear communication.

[CANDIDATE]'s strengths include:
• Data-driven decision making
• Creative problem solving
• Cross-functional collaboration
• Budget management and ROI optimization
• Stakeholder communication

I am confident that [CANDIDATE] will make an immediate and lasting impact in their next role. Any organization would be fortunate to have them on their team.

Please feel free to contact me for additional information.

Best regards,

[YOUR NAME]
[YOUR TITLE]
[COMPANY]
[CONTACT INFORMATION]"""},
        
        {"instruction": "Create a recommendation letter for a colleague applying for an executive MBA program.",
         "output": """[DATE]

Executive MBA Admissions Committee
[BUSINESS SCHOOL NAME]
[ADDRESS]

Dear Admissions Committee,

I am honored to recommend [CANDIDATE NAME] for your Executive MBA program. Having worked alongside [CANDIDATE] for [NUMBER] years at [COMPANY], I can attest to their exceptional leadership qualities and readiness for advanced business education.

As [CANDIDATE'S TITLE], [CANDIDATE] has demonstrated the strategic thinking and leadership acumen that your program seeks to develop. They currently oversee [SCOPE] with responsibility for [TEAM SIZE/BUDGET/FUNCTIONS].

Strategic Impact:
[CANDIDATE] led our [INITIATIVE], which resulted in [OUTCOME]. This required navigating complex organizational dynamics, building consensus across departments, and making difficult resource allocation decisions. The initiative ultimately generated $[AMOUNT] in value.

Leadership Excellence:
What distinguishes [CANDIDATE] is their ability to inspire and develop others. They have:
• Built and mentored a team of [NUMBER] professionals
• Championed diversity and inclusion initiatives
• Served as executive sponsor for [PROGRAM/INITIATIVE]

Intellectual Curiosity:
[CANDIDATE] constantly seeks to expand their knowledge and challenge assumptions. They regularly [EXAMPLES: lead book discussions, bring external perspectives, question conventional approaches].

Why An Executive MBA:
An Executive MBA will provide [CANDIDATE] with the frameworks and peer network to accelerate their trajectory toward [CAREER GOAL]. Your program's emphasis on [PROGRAM STRENGTH] aligns perfectly with their development needs.

I recommend [CANDIDATE] without reservation. They will contribute meaningfully to classroom discussions and represent your program exceptionally.

Sincerely,

[YOUR NAME]
[YOUR TITLE]
[COMPANY]
[CONTACT]"""},
    ]
    dataset.extend(recommendation_templates)
    
    # -------------------------------------------------------------------------
    # 7. TERMS OF SERVICE (15 variations)
    # -------------------------------------------------------------------------
    tos_templates = [
        {"instruction": "Write the introduction and acceptance section of Terms of Service for a web application.",
         "output": """TERMS OF SERVICE

Last Updated: [DATE]

Welcome to [SERVICE NAME]. These Terms of Service ("Terms") govern your access to and use of [COMPANY NAME]'s website, products, and services ("Services").

PLEASE READ THESE TERMS CAREFULLY BEFORE USING OUR SERVICES.

1. ACCEPTANCE OF TERMS

By accessing or using our Services, you agree to be bound by these Terms. If you disagree with any part of the Terms, you may not access the Services.

These Terms constitute a legally binding agreement between you and [COMPANY NAME] ("Company," "we," "us," or "our").

2. ELIGIBILITY

To use our Services, you must:
• Be at least 18 years of age, or the age of majority in your jurisdiction
• Have the legal capacity to enter into a binding agreement
• Not be prohibited from using the Services under applicable law

If you are using the Services on behalf of an organization, you represent that you have the authority to bind that organization to these Terms.

3. CHANGES TO TERMS

We reserve the right to modify these Terms at any time. We will provide notice of material changes by:
• Posting the updated Terms on our website
• Sending an email to registered users
• Displaying a notice within the Services

Your continued use of the Services after changes become effective constitutes acceptance of the revised Terms.

4. ADDITIONAL TERMS

Certain features may be subject to additional terms, which will be presented when you access those features. Such additional terms supplement these Terms and, in the event of conflict, the additional terms will control."""},
        
        {"instruction": "Draft the user responsibilities section of Terms of Service.",
         "output": """USER RESPONSIBILITIES AND ACCEPTABLE USE

1. Account Registration
You are responsible for:
• Providing accurate and complete registration information
• Maintaining the confidentiality of your login credentials
• All activities that occur under your account
• Notifying us immediately of any unauthorized use

2. Acceptable Use
You agree NOT to:
• Violate any applicable laws or regulations
• Infringe any intellectual property or privacy rights
• Transmit malware, viruses, or harmful code
• Attempt to gain unauthorized access to any systems
• Interfere with or disrupt the Services
• Engage in fraudulent or deceptive practices
• Harass, abuse, or harm other users
• Use the Services for any illegal purpose

3. Content Standards
Any content you submit must:
• Be accurate and not misleading
• Not be defamatory, obscene, or offensive
• Not infringe third-party rights
• Comply with all applicable laws

4. Prohibited Activities
The following are strictly prohibited:
• Automated data collection (scraping, crawling) without consent
• Creating multiple accounts for abusive purposes
• Circumventing security features or access controls
• Reverse engineering or decompiling our software
• Reselling or redistributing the Services without authorization

5. Consequences
Violation of these terms may result in:
• Warning and demand for compliance
• Temporary suspension of access
• Permanent termination of account
• Legal action where appropriate"""},
        
        {"instruction": "Create the intellectual property section of Terms of Service.",
         "output": """INTELLECTUAL PROPERTY RIGHTS

1. Our Intellectual Property
The Services and all content, features, and functionality, including but not limited to:
• Software, code, and algorithms
• Text, graphics, logos, and images
• Audio, video, and multimedia content
• User interface design and layout
• Trademarks, service marks, and trade dress

are owned by [COMPANY NAME], its licensors, or other providers and are protected by United States and international intellectual property laws.

2. Limited License
Subject to your compliance with these Terms, we grant you a limited, non-exclusive, non-transferable, revocable license to:
• Access and use the Services for personal or internal business purposes
• Download and display content solely as permitted by the Services

This license does not include:
• Modification or derivative works
• Commercial exploitation
• Data mining or extraction
• Framing or mirroring
• Linking except as expressly permitted

3. Your Content
You retain ownership of content you create and submit to the Services ("User Content"). By submitting User Content, you grant us a worldwide, non-exclusive, royalty-free license to:
• Host, store, and display your content
• Reproduce and modify as necessary to provide the Services
• Create derivative works for service improvement
• Use for promotional purposes (with your consent)

4. DMCA Notice
We respect intellectual property rights. To report infringement, contact our designated agent at [DMCA CONTACT EMAIL] with:
• Description of the copyrighted work
• Location of the infringing material
• Your contact information
• Statement of good faith belief
• Statement of accuracy under penalty of perjury"""},
        
        {"instruction": "Write the limitation of liability section for Terms of Service.",
         "output": """LIMITATION OF LIABILITY

1. DISCLAIMER OF WARRANTIES
THE SERVICES ARE PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO:
• MERCHANTABILITY
• FITNESS FOR A PARTICULAR PURPOSE
• NON-INFRINGEMENT
• ACCURACY OR COMPLETENESS
• UNINTERRUPTED OR ERROR-FREE OPERATION

WE DO NOT WARRANT THAT THE SERVICES WILL MEET YOUR REQUIREMENTS OR THAT DEFECTS WILL BE CORRECTED.

2. LIMITATION OF LIABILITY
TO THE MAXIMUM EXTENT PERMITTED BY LAW:

A) IN NO EVENT SHALL [COMPANY NAME], ITS DIRECTORS, EMPLOYEES, PARTNERS, AGENTS, OR AFFILIATES BE LIABLE FOR:
• Indirect, incidental, special, consequential, or punitive damages
• Loss of profits, revenue, data, or business opportunities
• Cost of substitute services
• Any damages arising from your use of the Services

B) OUR TOTAL LIABILITY FOR ANY CLAIMS ARISING FROM THESE TERMS OR YOUR USE OF THE SERVICES SHALL NOT EXCEED:
• The amount you paid us in the [12] months preceding the claim, OR
• [AMOUNT] dollars, whichever is greater

3. EXCEPTIONS
These limitations do not apply to:
• Liability that cannot be excluded by law
• Gross negligence or willful misconduct
• Death or personal injury caused by our negligence
• Fraud or fraudulent misrepresentation

4. JURISDICTIONAL LIMITATIONS
Some jurisdictions do not allow certain warranty disclaimers or liability limitations. In such jurisdictions, our liability is limited to the maximum extent permitted by law.

5. BASIS OF THE BARGAIN
You acknowledge that we have set our prices and entered into these Terms in reliance upon the limitations of liability and disclaimers herein, which form an essential basis of the bargain between us."""},
    ]
    dataset.extend(tos_templates)
    
    # -------------------------------------------------------------------------
    # 8. PROPOSALS AND EXECUTIVE SUMMARIES (15 variations)
    # -------------------------------------------------------------------------
    proposal_templates = [
        {"instruction": "Write an executive summary for a software development proposal.",
         "output": """EXECUTIVE SUMMARY

Proposal: [PROJECT NAME] Development
Prepared for: [CLIENT COMPANY]
Prepared by: [YOUR COMPANY]
Date: [DATE]

OVERVIEW
[YOUR COMPANY] proposes to design and develop [PROJECT DESCRIPTION] to address [CLIENT COMPANY]'s need for [BUSINESS NEED]. This solution will enable [KEY BENEFITS] and position [CLIENT COMPANY] for [STRATEGIC OUTCOME].

THE CHALLENGE
[CLIENT COMPANY] currently faces:
• [CHALLENGE 1]
• [CHALLENGE 2]
• [CHALLENGE 3]

These challenges result in [QUANTIFIED IMPACT: lost revenue, inefficiency, etc.].

OUR SOLUTION
We propose a [SOLUTION TYPE] that will:
• [CAPABILITY 1] - enabling [BENEFIT]
• [CAPABILITY 2] - resulting in [BENEFIT]
• [CAPABILITY 3] - providing [BENEFIT]

KEY FEATURES
• [FEATURE 1]: [Brief description]
• [FEATURE 2]: [Brief description]
• [FEATURE 3]: [Brief description]

PROJECTED OUTCOMES
Based on similar implementations, we anticipate:
• [METRIC]% improvement in [AREA]
• $[AMOUNT] in annual cost savings
• [NUMBER]-month payback period
• ROI of [PERCENTAGE]% over [TIMEFRAME]

INVESTMENT AND TIMELINE
• Total Investment: $[AMOUNT]
• Duration: [NUMBER] months
• Go-Live Date: [DATE]

WHY [YOUR COMPANY]
• [NUMBER]+ years of experience in [INDUSTRY]
• Proven track record with [SIMILAR CLIENTS]
• [CERTIFICATION/PARTNERSHIP] credentials
• Dedicated support and maintenance

We are confident this solution will deliver significant value to [CLIENT COMPANY]. We welcome the opportunity to discuss this proposal in detail.

[YOUR NAME], [TITLE]
[CONTACT INFORMATION]"""},
        
        {"instruction": "Create a consulting services proposal introduction.",
         "output": """PROPOSAL FOR CONSULTING SERVICES

[PROJECT/ENGAGEMENT NAME]

Submitted to:
[CLIENT NAME]
[TITLE]
[CLIENT COMPANY]

Submitted by:
[YOUR COMPANY]
[ADDRESS]

Date: [DATE]
Proposal Valid Until: [DATE]

---

INTRODUCTION

Dear [CLIENT NAME],

Thank you for the opportunity to submit this proposal for [ENGAGEMENT TYPE] consulting services. Following our meeting on [DATE], we have developed a comprehensive approach to address your organization's needs.

[YOUR COMPANY] is a [DESCRIPTION] consulting firm specializing in [AREAS OF EXPERTISE]. Since [YEAR], we have helped [NUMBER]+ organizations achieve [OUTCOMES].

UNDERSTANDING YOUR NEEDS
Based on our discussions, we understand that [CLIENT COMPANY] is seeking to:
1. [OBJECTIVE 1]
2. [OBJECTIVE 2]
3. [OBJECTIVE 3]

You have identified the following key challenges:
• [CHALLENGE/PAIN POINT 1]
• [CHALLENGE/PAIN POINT 2]
• [CHALLENGE/PAIN POINT 3]

OUR APPROACH
We propose a [PHASED/COMPREHENSIVE] approach that:
• Builds upon [CLIENT COMPANY]'s existing strengths
• Addresses immediate priorities while establishing long-term capabilities
• Ensures knowledge transfer to your team
• Delivers measurable, sustainable results

This proposal outlines our methodology, team, timeline, and investment. We are committed to working collaboratively with your team to achieve your objectives.

We look forward to the opportunity to partner with [CLIENT COMPANY] on this important initiative."""},
        
        {"instruction": "Draft a project scope statement for a website redesign.",
         "output": """PROJECT SCOPE STATEMENT

Project Name: [CLIENT] Website Redesign
Project Manager: [NAME]
Date: [DATE]
Version: [NUMBER]

1. PROJECT OBJECTIVES
The primary objectives of this project are to:
• Redesign [CLIENT]'s corporate website to improve user experience
• Implement modern, responsive design supporting all devices
• Improve site performance and page load times
• Enhance SEO capabilities and organic search visibility
• Integrate with [CRM/MARKETING AUTOMATION SYSTEM]

2. SCOPE DESCRIPTION

In Scope:
• Discovery and requirements gathering
• UX research and user journey mapping
• Visual design (up to [NUMBER] unique page templates)
• Responsive front-end development
• CMS implementation ([PLATFORM])
• Content migration (up to [NUMBER] pages)
• Basic SEO optimization
• Quality assurance and testing
• Deployment and launch support
• Post-launch support ([NUMBER] weeks)

Out of Scope:
• Content creation or copywriting
• Photography or video production
• E-commerce functionality
• Mobile application development
• Third-party integrations beyond [SPECIFIED SYSTEMS]
• Ongoing maintenance and hosting
• Multi-language versions

3. DELIVERABLES
• Requirements documentation
• Sitemap and information architecture
• Wireframes for [NUMBER] key pages
• Visual design mockups
• Functional website on [PLATFORM]
• Training documentation
• Analytics setup

4. ASSUMPTIONS
• Client will provide all content within agreed timelines
• Client will designate a primary point of contact
• Existing brand guidelines will be provided
• [NUMBER] rounds of revision are included per phase

5. CONSTRAINTS
• Budget: $[AMOUNT]
• Timeline: [NUMBER] weeks
• Must launch before [DATE/EVENT]"""},
    ]
    dataset.extend(proposal_templates)
    
    # -------------------------------------------------------------------------
    # 9. SERVICE AGREEMENTS (10 variations)
    # -------------------------------------------------------------------------
    service_templates = [
        {"instruction": "Write the scope of services section for a software maintenance agreement.",
         "output": """SCOPE OF SERVICES

1. COVERED SERVICES
Under this Software Maintenance Agreement, Provider shall deliver the following services:

1.1 Technical Support
• Help desk support via email and phone
• Support hours: [HOURS], [DAYS]
• Initial response times:
  - Critical issues: [X] hours
  - High priority: [X] hours
  - Medium priority: [X] business hours
  - Low priority: [X] business days

1.2 Bug Fixes and Patches
• Investigation and resolution of software defects
• Security patches and vulnerability fixes
• Critical patches deployed within [X] hours
• Standard patches included in monthly releases

1.3 Software Updates
• [NUMBER] minor releases per year
• [NUMBER] major releases per year
• Release notes and documentation
• Backward compatibility for [NUMBER] versions

1.4 Preventive Maintenance
• System health monitoring
• Performance optimization recommendations
• Database maintenance and optimization
• Log analysis and review

2. SERVICE EXCLUSIONS
The following are not covered under this Agreement:
• Custom development or feature requests
• Hardware maintenance or replacement
• Third-party software support
• Data recovery from user error
• Support for modified or customized software
• Training beyond standard documentation

3. SERVICE LEVELS
| Priority | Response Time | Resolution Target |
|----------|---------------|-------------------|
| Critical | 1 hour | 4 hours |
| High | 4 hours | 8 hours |
| Medium | 8 hours | 3 business days |
| Low | 24 hours | 5 business days |

Service credits apply for missed targets as specified in Section [X]."""},
        
        {"instruction": "Create the payment and fees section of a professional services agreement.",
         "output": """FEES AND PAYMENT TERMS

1. SERVICE FEES

1.1 Fixed Fee Projects
The total fee for fixed-scope projects shall be as specified in the applicable Statement of Work ("SOW"). Fees shall be paid according to the milestone schedule defined in the SOW.

1.2 Time and Materials
For time and materials engagements:
• Hourly rates by role are specified in Exhibit A
• Rates are valid for [12] months from the Effective Date
• Rate increases shall not exceed [X]% annually

1.3 Retainer Services
• Monthly retainer: $[AMOUNT]
• Includes up to [NUMBER] hours per month
• Unused hours do not roll over
• Additional hours billed at $[RATE]/hour

2. EXPENSES
Reasonable, pre-approved expenses shall be reimbursed at cost:
• Travel: Economy class airfare, standard hotel accommodations
• Meals: Per diem of $[AMOUNT] per day
• Other: Receipts required for expenses over $[AMOUNT]

Total reimbursable expenses shall not exceed [X]% of professional fees without prior approval.

3. INVOICING AND PAYMENT

3.1 Invoicing Schedule
• Fixed fee: Per milestone completion
• Time and materials: Bi-weekly or monthly
• Retainer: First of each month in advance

3.2 Payment Terms
• Payment due: Net [30] days from invoice date
• Late payment fee: [1.5]% per month
• Payments via [ACH/CHECK/WIRE]

3.3 Disputed Invoices
Client shall notify Provider of disputes within [15] days. Undisputed amounts remain due per standard terms.

4. TAXES
Fees are exclusive of applicable taxes. Client shall pay all sales, use, and similar taxes."""},
    ]
    dataset.extend(service_templates)
    
    # -------------------------------------------------------------------------
    # 10. ADDITIONAL BUSINESS DOCUMENTS (20+ variations)
    # -------------------------------------------------------------------------
    additional_templates = [
        {"instruction": "Write a press release announcing a company acquisition.",
         "output": """FOR IMMEDIATE RELEASE

[ACQUIRING COMPANY] COMPLETES ACQUISITION OF [TARGET COMPANY]

Strategic combination creates [INDUSTRY] leader with enhanced capabilities

[CITY, STATE] – [DATE] – [ACQUIRING COMPANY] (NASDAQ: [TICKER]) today announced the completion of its acquisition of [TARGET COMPANY], a leading provider of [TARGET'S BUSINESS DESCRIPTION].

The transaction, valued at approximately $[AMOUNT], was approved by shareholders of both companies and all regulatory bodies. The combined company will operate under the [ACQUIRING/NEW] brand.

"This acquisition marks a transformative moment for our company," said [CEO NAME], Chief Executive Officer of [ACQUIRING COMPANY]. "By combining [ACQUIRING COMPANY]'s [STRENGTH] with [TARGET COMPANY]'s [STRENGTH], we are positioned to deliver unprecedented value to our customers and shareholders."

Transaction Highlights:
• Creates a [INDUSTRY] leader with combined revenues of $[AMOUNT]
• Expands geographic presence to [NUMBER] countries
• Brings together [NUMBER] employees worldwide
• Expected to generate $[AMOUNT] in annual synergies

[TARGET CEO NAME], former CEO of [TARGET COMPANY], commented: "Joining forces with [ACQUIRING COMPANY] allows us to accelerate our mission to [MISSION]. Our combined expertise will drive innovation and growth."

The acquisition is expected to be immediately accretive to earnings and create significant long-term shareholder value.

About [ACQUIRING COMPANY]
[BRIEF COMPANY DESCRIPTION]

Media Contact:
[NAME]
[EMAIL]
[PHONE]

Investor Contact:
[NAME]
[EMAIL]
[PHONE]

###"""},
        
        {"instruction": "Draft a company policy on data classification and handling.",
         "output": """DATA CLASSIFICATION AND HANDLING POLICY

Policy Number: [NUMBER]
Effective Date: [DATE]
Last Reviewed: [DATE]
Owner: [DEPARTMENT/TITLE]

1. PURPOSE
This policy establishes a framework for classifying and handling company data based on sensitivity and criticality. Proper classification ensures appropriate protection and handling of information assets.

2. SCOPE
This policy applies to all employees, contractors, and third parties who access, process, or store company data in any format.

3. DATA CLASSIFICATION LEVELS

3.1 PUBLIC
• Information intended for public release
• Marketing materials, press releases
• No special handling required

3.2 INTERNAL
• Information for internal use only
• Operational procedures, internal communications
• Requires: Access controls, secure disposal

3.3 CONFIDENTIAL
• Sensitive business information
• Financial data, strategic plans, customer information
• Requires: Encryption, access logging, NDA

3.4 RESTRICTED
• Highest sensitivity data
• Trade secrets, PII, regulated data (HIPAA, PCI)
• Requires: Strong encryption, strict access controls, audit trails

4. HANDLING REQUIREMENTS

| Classification | Storage | Transmission | Disposal |
|----------------|---------|--------------|----------|
| Public | Any | Any | Standard |
| Internal | Approved systems | Secure methods | Shredding |
| Confidential | Encrypted storage | Encryption required | Certified destruction |
| Restricted | Dedicated systems | End-to-end encryption | Witnessed destruction |

5. LABELING
All documents and files must be clearly labeled with their classification level.

6. COMPLIANCE
Violations may result in disciplinary action up to and including termination.

7. REVIEW
This policy shall be reviewed annually and updated as needed."""},
        
        {"instruction": "Create an employee handbook section on workplace conduct.",
         "output": """WORKPLACE CONDUCT AND PROFESSIONALISM

1. PROFESSIONAL STANDARDS
All employees are expected to conduct themselves in a professional manner that reflects positively on themselves and [COMPANY NAME].

1.1 General Expectations
• Treat colleagues, customers, and partners with respect and courtesy
• Communicate openly, honestly, and constructively
• Take responsibility for your actions and decisions
• Support a positive and inclusive work environment
• Represent the company professionally in all interactions

1.2 Dress Code
[COMPANY NAME] maintains a [business casual/professional] dress code. Employees should dress appropriately for their role and any scheduled meetings or events. Department managers may establish specific guidelines for their teams.

2. WORKPLACE BEHAVIOR

2.1 Prohibited Conduct
The following behaviors are strictly prohibited:
• Harassment or discrimination of any kind
• Violence, threats, or intimidating behavior
• Theft, fraud, or dishonesty
• Unauthorized disclosure of confidential information
• Possession of weapons on company premises
• Being under the influence of alcohol or illegal drugs

2.2 Conflicts of Interest
Employees must avoid situations where personal interests conflict with company interests. Any potential conflicts must be disclosed to Human Resources immediately.

2.3 Use of Company Resources
Company resources, including equipment, supplies, and time, should be used for legitimate business purposes. Limited personal use may be permitted per department guidelines.

3. DIGITAL CONDUCT
• Use professional language in all electronic communications
• Do not send harassing, discriminatory, or offensive content
• Be mindful that company communications may be monitored
• Represent the company appropriately on social media

4. REPORTING CONCERNS
Employees should report conduct violations to their manager, Human Resources, or through the anonymous ethics hotline.

5. CONSEQUENCES
Violations of this policy may result in disciplinary action up to and including termination of employment."""},

        {"instruction": "Write a vendor evaluation criteria document.",
         "output": """VENDOR EVALUATION CRITERIA

Document: Vendor Selection Scorecard
Project: [PROJECT NAME]
Date: [DATE]
Prepared by: [NAME]

1. EVALUATION OVERVIEW
This document establishes the criteria and weighting for evaluating vendors for [PRODUCT/SERVICE]. All vendors will be scored using these standardized criteria to ensure objective comparison.

2. EVALUATION CATEGORIES AND WEIGHTS

| Category | Weight | Description |
|----------|--------|-------------|
| Technical Capabilities | 30% | Product functionality and fit |
| Cost and Value | 25% | Total cost of ownership and ROI |
| Vendor Viability | 20% | Financial stability and market position |
| Implementation | 15% | Deployment approach and timeline |
| Support and Service | 10% | Ongoing support capabilities |

3. DETAILED CRITERIA

3.1 Technical Capabilities (30%)
• Functional requirements coverage (1-5 scale)
• Technical architecture alignment
• Integration capabilities
• Scalability and performance
• Security and compliance features
• Innovation and roadmap

3.2 Cost and Value (25%)
• Initial licensing/purchase costs
• Implementation costs
• Ongoing maintenance and support fees
• Hidden costs (training, customization)
• Total cost of ownership (3-year)
• Expected ROI

3.3 Vendor Viability (20%)
• Financial stability (revenue, profitability)
• Market presence and reputation
• Customer base and references
• Industry expertise
• Strategic direction alignment

3.4 Implementation (15%)
• Proposed methodology
• Timeline and milestones
• Resource requirements
• Risk mitigation approach
• Change management support

3.5 Support and Service (10%)
• Support availability and channels
• SLA commitments
• Training offerings
• Documentation quality
• Customer success resources

4. SCORING METHODOLOGY
Each criterion scored 1-5:
5 = Exceeds requirements
4 = Fully meets requirements
3 = Meets most requirements
2 = Partially meets requirements
1 = Does not meet requirements

5. MINIMUM THRESHOLDS
Vendors must achieve:
• Overall score: minimum 3.5
• Technical capabilities: minimum 3.0
• No individual criterion below 2.0"""},
    ]
    dataset.extend(additional_templates)
    
    return dataset


# ============================================================================
# DATASET PROCESSING
# ============================================================================

def create_tokenized_dataset(dataset: List[Dict], tokenizer, max_length: int = 512):
    """Create a tokenized HuggingFace dataset."""
    
    # Format data for training
    formatted_data = []
    for item in dataset:
        text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
        formatted_data.append({"text": text})
    
    # Create HuggingFace dataset
    hf_dataset = Dataset.from_list(formatted_data)
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = hf_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_model_and_tokenizer(model_name: str, lora_r: int, lora_alpha: int, device: str):
    """Load the base model and tokenizer with LoRA configuration."""
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    # Configure LoRA with specified parameters
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "c_fc"],  # Added c_fc for more capacity
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA Configuration: r={lora_r}, alpha={lora_alpha}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"Total parameters: {total_params:,}")
    
    return model, tokenizer


# ============================================================================
# TRAINING
# ============================================================================

def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gradient_accumulation: int,
    warmup_ratio: float,
    use_early_stopping: bool = True,
):
    """Train the model with comprehensive logging and evaluation."""
    logger.info("Setting up training arguments...")
    
    # Calculate steps for logging
    total_steps = (len(train_dataset) // (batch_size * gradient_accumulation)) * epochs
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        
        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        logging_first_step=True,
        report_to=["tensorboard"],
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=50,
        
        # Saving
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Optimization
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        
        # Other
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        seed=42,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Setup callbacks
    callbacks = []
    if use_early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=5))
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Total training steps: ~{total_steps}")
    logger.info(f"Evaluation every 50 steps, checkpoints every 100 steps")
    
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training config
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gradient_accumulation": gradient_accumulation,
        "final_train_loss": trainer.state.log_history[-1].get("loss", "N/A"),
        "final_eval_loss": trainer.state.log_history[-1].get("eval_loss", "N/A"),
    }
    with open(f"{output_dir}/training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("Training complete!")
    return trainer


def generate_text(model, tokenizer, instruction: str, max_length: int = 512, device: str = "cuda"):
    """Generate text given an instruction."""
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:\n" in generated_text:
        response = generated_text.split("### Response:\n")[-1].strip()
    else:
        response = generated_text
    
    return response


def test_generation(model, tokenizer, device: str = "cuda"):
    """Test the model with sample instructions."""
    test_instructions = [
        "Write a confidentiality clause for a software development contract.",
        "Draft a professional email requesting a deadline extension from a client.",
        "Create meeting minutes for a weekly project status meeting.",
        "Write the introduction of a mutual non-disclosure agreement.",
        "Draft a termination clause for an employment agreement.",
    ]
    
    logger.info("\n" + "=" * 60)
    logger.info("TESTING MODEL GENERATION")
    logger.info("=" * 60)
    
    for i, instruction in enumerate(test_instructions, 1):
        logger.info(f"\n--- Test {i} ---")
        logger.info(f"Instruction: {instruction}")
        
        response = generate_text(model, tokenizer, instruction, device=device)
        
        logger.info(f"\nGenerated Response:\n{response[:1500]}...")
        logger.info("-" * 40)


# ============================================================================
# MAIN
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 for legal/business text - LONG QUALITY TRAINING",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--preset", type=str, choices=list(TRAINING_PRESETS.keys()),
                        help="Use a predefined training configuration")
    parser.add_argument("--model_name", type=str, default="gpt2-medium",
                        help="Base model (gpt2, gpt2-medium, gpt2-large)")
    parser.add_argument("--output_dir", type=str, default="./legal_llm_finetuned",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs")
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--eval_split", type=float, default=0.1,
                        help="Fraction of data for evaluation")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--no_early_stopping", action="store_true",
                        help="Disable early stopping")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Apply preset if specified
    if args.preset:
        preset = TRAINING_PRESETS[args.preset]
        logger.info(f"Using preset: {args.preset} - {preset['description']}")
        args.epochs = preset["epochs"]
        args.lora_r = preset["lora_r"]
        args.lora_alpha = preset["lora_alpha"]
        args.learning_rate = preset["learning_rate"]
        args.batch_size = preset["batch_size"]
        args.gradient_accumulation = preset["gradient_accumulation"]
        args.warmup_ratio = preset["warmup_ratio"]
        args.eval_split = preset["eval_split"]
    
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 60)
    logger.info("LEGAL/BUSINESS LLM FINE-TUNING - QUALITY TRAINING")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Log training configuration
    logger.info("\nTraining Configuration:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  LoRA rank: {args.lora_r}")
    logger.info(f"  LoRA alpha: {args.lora_alpha}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Gradient accumulation: {args.gradient_accumulation}")
    logger.info(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    logger.info(f"  Warmup ratio: {args.warmup_ratio}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset
    logger.info("\nCreating legal/business text dataset...")
    dataset = create_legal_business_dataset()
    logger.info(f"Total dataset size: {len(dataset)} examples")
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name, args.lora_r, args.lora_alpha, device
    )
    
    # Tokenize and split dataset
    logger.info("Tokenizing dataset...")
    full_dataset = create_tokenized_dataset(dataset, tokenizer, args.max_length)
    
    # Split into train/eval
    split = full_dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # Train
    trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation=args.gradient_accumulation,
        warmup_ratio=args.warmup_ratio,
        use_early_stopping=not args.no_early_stopping,
    )
    
    # Test generation
    logger.info("\nTesting model generation...")
    test_generation(model, tokenizer, device)
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"TensorBoard logs: {args.output_dir}/logs")
    logger.info("\nTo view training curves:")
    logger.info(f"  tensorboard --logdir {args.output_dir}/logs")
    logger.info("\nTo test the model:")
    logger.info(f"  python app.py --model_path {args.output_dir}")


if __name__ == "__main__":
    main()

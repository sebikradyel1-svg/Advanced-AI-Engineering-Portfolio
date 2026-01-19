#!/usr/bin/env python3
"""
English Legal/Business LLM Fine-Tuning Script
==============================================

Fine-tunes GPT-2 base model using LoRA/PEFT for generating English legal and
business text including contracts, NDAs, policies, and professional correspondence.

Optimized for RTX 3060 6GB VRAM with gradient accumulation and mixed precision.

Usage:
    # Full training run
    python english_legal_llm_finetuning.py --output_dir ./legal_llm_model --epochs 3
    
    # Test mode (quick verification with 1 epoch, small subset)
    python english_legal_llm_finetuning.py --test_only --epochs 1
    
    # Custom configuration
    python english_legal_llm_finetuning.py --model_name gpt2 --batch_size 2 --epochs 5

Author: AI Engineering Portfolio
License: MIT
"""

import argparse
import logging
import os
import sys
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
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
# DATASET: English Legal/Business Text Templates (100+ examples)
# ============================================================================

def create_legal_business_dataset() -> List[Dict[str, str]]:
    """
    Creates a comprehensive dataset of English legal and business text templates.
    Contains 100+ diverse examples across 10 template categories.
    """
    
    dataset = []
    
    # -------------------------------------------------------------------------
    # 1. EMPLOYMENT CONTRACTS (12 variations)
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
The Company reserves the right to revoke remote work authorization with reasonable notice."""},
        
        {"instruction": "Create a probationary period clause for a new employee contract.",
         "output": """PROBATIONARY PERIOD

1. Duration
Employee shall serve an initial probationary period of [NUMBER] days from the Start Date.

2. Purpose
The Probationary Period assesses Employee's suitability for the position, including performance, skills, and cultural fit.

3. Performance Reviews
Employee shall receive formal feedback at 30-day, 60-day, and 90-day intervals.

4. Termination During Probation
During the Probationary Period, either party may terminate with [NUMBER] days' notice. Termination during probation shall not give rise to severance benefits.

5. Extension
The Company may extend the Probationary Period for an additional [NUMBER] days if needed.

6. Successful Completion
Upon successful completion, Employee becomes a regular employee entitled to all benefits described in this Agreement."""},
        
        {"instruction": "Write the governing law section of an employment contract.",
         "output": """GOVERNING LAW AND DISPUTE RESOLUTION

1. Governing Law
This Agreement shall be governed by the laws of the State of [STATE], without regard to conflict of laws principles.

2. Mandatory Arbitration
Any dispute arising from this Agreement shall be resolved by binding arbitration in [CITY, STATE] before one arbitrator, administered by JAMS pursuant to its Employment Arbitration Rules.

3. Costs and Fees
The Company shall pay arbitration fees beyond what Employee would pay in court. Each party bears its own attorneys' fees unless the arbitrator determines otherwise.

4. Confidentiality
All arbitration proceedings and awards shall remain confidential.

5. Equitable Relief
Either party may seek injunctive relief in court to prevent irreparable harm pending arbitration.

6. WAIVER OF JURY TRIAL
EACH PARTY IRREVOCABLY WAIVES ANY RIGHT TO TRIAL BY JURY."""},
        
        {"instruction": "Draft a relocation assistance clause for an employment contract.",
         "output": """RELOCATION ASSISTANCE

1. Relocation Package
The Company shall provide relocation assistance for Employee's move from [ORIGIN] to [DESTINATION].

2. Covered Expenses
The Company shall reimburse the following up to $[AMOUNT]:
   a) Moving costs for household goods
   b) Temporary housing for up to [NUMBER] days
   c) House-hunting trip expenses
   d) Travel expenses to new location

3. Timing
Relocation must be completed within [NUMBER] days of the Start Date.

4. Repayment Obligation
If Employee voluntarily resigns or is terminated for Cause within [NUMBER] months, Employee shall repay relocation benefits:
   - Within 12 months: 100% repayment
   - 12-18 months: 66% repayment
   - 18-24 months: 33% repayment

5. Documentation
Employee shall submit itemized receipts within 30 days of incurring expenses."""},
        
        {"instruction": "Write a comprehensive employment contract for a Chief Technology Officer position.",
         "output": """EXECUTIVE EMPLOYMENT AGREEMENT - CHIEF TECHNOLOGY OFFICER

This Agreement is effective as of [DATE] between GlobalTech Industries, Inc. ("Company") and [NAME] ("Executive").

1. POSITION
Executive shall serve as Chief Technology Officer, reporting to the CEO and serving on the Executive Leadership Team.

2. DUTIES
Executive shall: (a) lead technology strategy and innovation; (b) oversee engineering and IT teams; (c) manage technology budget; (d) ensure cybersecurity compliance.

3. COMPENSATION
   Base Salary: $[AMOUNT] per year
   Annual Bonus: Target of [PERCENTAGE]% of base salary
   Equity: [NUMBER] restricted stock units vesting over 4 years
   Signing Bonus: $[AMOUNT]

4. TERM
Initial term of [NUMBER] years, automatically renewing unless terminated.

5. SEVERANCE
Upon termination without Cause: (a) [NUMBER] months' salary; (b) pro-rated bonus; (c) COBRA coverage; (d) accelerated vesting.

6. CHANGE OF CONTROL
Upon qualifying termination within 12 months of Change of Control: 100% acceleration of unvested equity."""},
        
        {"instruction": "Create an internship agreement template for a summer legal internship.",
         "output": """LEGAL INTERNSHIP AGREEMENT

This Agreement is between [LAW FIRM NAME] ("Firm") and [INTERN NAME] ("Intern").

1. POSITION
Position: Summer Legal Intern
Duration: [START DATE] to [END DATE]
Hours: Full-time, 40 hours per week

2. COMPENSATION
Intern shall receive a weekly stipend of $[AMOUNT]. This internship does not constitute employment.

3. RESPONSIBILITIES
Intern shall: (a) conduct legal research; (b) draft memoranda and briefs; (c) attend client meetings when appropriate; (d) participate in training programs.

4. SUPERVISION
Intern shall report to [SUPERVISOR NAME], who shall provide mentorship and performance evaluations.

5. CONFIDENTIALITY
Intern acknowledges the confidential nature of client information and agrees to maintain strict confidentiality.

6. PROFESSIONAL CONDUCT
Intern shall comply with all Firm policies and applicable rules of professional conduct.

7. ACADEMIC CREDIT
Intern may seek academic credit. The Firm shall cooperate with reasonable documentation requests."""},
    ]
    
    # -------------------------------------------------------------------------
    # 2. NON-DISCLOSURE AGREEMENTS (12 variations)
    # -------------------------------------------------------------------------
    nda_templates = [
        {"instruction": "Write a mutual non-disclosure agreement introduction for a business partnership.",
         "output": """MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement ("Agreement") is entered into as of [DATE] by and between:

[COMPANY A], a [STATE] corporation ("Party A")
and
[COMPANY B], a [STATE] corporation ("Party B")

RECITALS

WHEREAS, the Parties wish to explore a potential business relationship regarding [PURPOSE];

WHEREAS, each Party may disclose confidential and proprietary information;

WHEREAS, the Parties desire to protect such confidential information;

NOW, THEREFORE, in consideration of the mutual promises contained herein, the Parties agree as follows:"""},
        
        {"instruction": "Draft the definition of Confidential Information for a technology company NDA.",
         "output": """DEFINITION OF CONFIDENTIAL INFORMATION

1. "Confidential Information" means any non-public information disclosed by the Disclosing Party, including:

   a) Technical Information: source code, algorithms, software architecture, APIs, technical specifications, product roadmaps, research data, and prototypes;

   b) Business Information: financial data, pricing models, customer lists, marketing strategies, business plans, and operational procedures;

   c) Personnel Information: employee records and compensation data;

   d) Legal Information: pending litigation, intellectual property filings, and contractual arrangements.

2. Confidential Information also includes any analyses or derivative works reflecting Confidential Information.

3. Exclusions: Information that (a) was publicly available prior to disclosure; (b) becomes public through no fault of the Receiving Party; (c) was independently developed; (d) was rightfully obtained from a third party without restriction."""},
        
        {"instruction": "Write the obligations section of a non-disclosure agreement.",
         "output": """OBLIGATIONS AND RESTRICTIONS

1. Confidentiality Obligations
The Receiving Party agrees to:
   a) Protect Confidential Information using at least the same degree of care as its own confidential information;
   b) Use Confidential Information solely for the stated Purpose;
   c) Restrict disclosure to employees and advisors with a need to know;
   d) Promptly notify of any unauthorized use or disclosure.

2. Prohibited Activities
The Receiving Party shall not:
   a) Copy or reproduce except as necessary;
   b) Reverse engineer any products containing Confidential Information;
   c) Remove proprietary notices;
   d) Use to compete with the Disclosing Party.

3. Required Disclosures
If compelled by law, the Receiving Party shall: (a) provide prompt notice; (b) cooperate to obtain protective orders; (c) disclose only minimum required information."""},
        
        {"instruction": "Create a one-way NDA for sharing information with a potential vendor.",
         "output": """ONE-WAY NON-DISCLOSURE AGREEMENT

This Agreement is made as of [DATE] between [COMPANY NAME] ("Discloser") and [VENDOR NAME] ("Recipient").

1. PURPOSE
Discloser intends to share confidential information for evaluating a potential vendor relationship.

2. CONFIDENTIAL INFORMATION
Includes: technical specifications, system requirements, security protocols, pricing requirements, and business processes.

3. RECIPIENT'S OBLIGATIONS
Recipient agrees to:
   a) Maintain strict confidentiality;
   b) Use information only for preparing a proposal;
   c) Limit access to authorized personnel;
   d) Not disclose to third parties;
   e) Return or destroy upon request.

4. EXCLUSIONS
Obligations do not apply to information that becomes public through no breach or was independently developed.

5. TERM
This Agreement remains effective for [NUMBER] years.

6. REMEDIES
Recipient acknowledges breach may cause irreparable harm, entitling Discloser to injunctive relief."""},
        
        {"instruction": "Write an NDA clause for employee invention assignment.",
         "output": """INVENTION ASSIGNMENT AND DISCLOSURE AGREEMENT

As a condition of employment, Employee agrees:

1. DISCLOSURE OBLIGATION
Employee shall promptly disclose in writing all Inventions conceived during employment.

"Inventions" means all discoveries, ideas, designs, developments, improvements, trade secrets, formulas, processes, and know-how.

2. ASSIGNMENT OF INVENTIONS
Employee assigns to the Company all right, title, and interest in Inventions that:
   a) Relate to the Company's business or development;
   b) Result from work performed for the Company;
   c) Were developed using Company resources.

3. EXCLUSIONS
This does not apply to Inventions developed entirely on Employee's own time without Company resources.

4. PRIOR INVENTIONS
Employee has listed in Exhibit A all prior inventions to be excluded.

5. COOPERATION
Employee agrees to assist in obtaining patents and protections for assigned Inventions.

6. WORKS FOR HIRE
All copyrightable works created in scope of employment are "works made for hire" owned by the Company."""},
        
        {"instruction": "Draft an NDA for M&A due diligence.",
         "output": """CONFIDENTIALITY AGREEMENT FOR M&A DUE DILIGENCE

This Agreement is entered into between [TARGET COMPANY] ("Target") and [ACQUIRING COMPANY] ("Acquirer") regarding a potential acquisition ("Transaction").

1. CONFIDENTIAL INFORMATION
Includes: financial statements, tax records, contracts, customer data, intellectual property, employee information, litigation matters, and Transaction-related materials.

2. USE RESTRICTION
Acquirer shall use Confidential Information solely for evaluating the Transaction. If not consummated, Acquirer shall not use any information.

3. PERMITTED DISCLOSURE
Acquirer may disclose only to directors, officers, employees, attorneys, and financial advisors ("Representatives") who agree to maintain confidentiality.

4. STANDSTILL
For [NUMBER] months, Acquirer shall not acquire Target securities, make public announcements, or solicit proxies without consent.

5. NON-SOLICITATION
For [NUMBER] months after termination of discussions, Acquirer shall not solicit Target employees.

6. RETURN OF MATERIALS
Upon request or termination, Acquirer shall return or destroy all Confidential Information and certify destruction."""},
        
        {"instruction": "Create an NDA for software beta testing participants.",
         "output": """BETA TESTER NON-DISCLOSURE AGREEMENT

This Agreement is between [SOFTWARE COMPANY] ("Company") and the undersigned ("Beta Tester").

1. BETA PROGRAM
Beta Tester is selected to test [SOFTWARE NAME] ("Beta Software"). Participation is voluntary.

2. CONFIDENTIAL INFORMATION
The following is confidential:
   - The Beta Software and documentation
   - Features and user interface designs
   - Performance data and bug reports
   - Release dates and roadmaps
   - Other testers' identities
   - The beta program's existence

3. OBLIGATIONS
Beta Tester agrees to:
   a) Keep all information strictly confidential
   b) Not share screenshots, videos, or descriptions
   c) Not discuss on social media or forums
   d) Use only for testing purposes
   e) Report feedback through designated channels

4. NO REVERSE ENGINEERING
Beta Tester shall not reverse engineer or decompile the Beta Software.

5. FEEDBACK
All feedback becomes Company property.

6. NO WARRANTY
Beta Software is provided "AS IS" without warranty.

7. TERM
Continues until public release or written release from obligations."""},
        
        {"instruction": "Write a confidentiality clause for customer data shared with service providers.",
         "output": """DATA PROCESSING AND CONFIDENTIALITY ADDENDUM

This Addendum supplements the Master Services Agreement between [CUSTOMER] and [PROVIDER].

1. CUSTOMER DATA
"Customer Data" means all data provided by Customer or collected by Provider on Customer's behalf.

2. CONFIDENTIALITY
Provider shall:
   a) Treat all Customer Data as Confidential Information
   b) Process only as necessary to provide Services
   c) Not sell or disclose to third parties
   d) Implement appropriate security measures
   e) Notify Customer of any data breach

3. SECURITY REQUIREMENTS
Provider shall maintain:
   - Encryption in transit and at rest
   - Access controls
   - Regular security assessments
   - Information security policies
   - Employee training

4. SUBPROCESSORS
Provider shall not engage subprocessors without prior written consent.

5. DATA SUBJECT RIGHTS
Provider shall assist Customer in responding to data subject requests.

6. AUDIT RIGHTS
Customer may audit Provider's compliance.

7. DATA RETURN/DELETION
Upon termination, Provider shall return or delete all Customer Data within [NUMBER] days."""},
        
        {"instruction": "Draft an NDA for joint research and development collaboration.",
         "output": """JOINT R&D CONFIDENTIALITY AGREEMENT

This Agreement is between [PARTY A] and [PARTY B] effective as of [DATE].

1. PURPOSE
The Parties will engage in joint research and development regarding [TECHNOLOGY AREA] ("Project").

2. BACKGROUND IP
Each Party retains ownership of its pre-existing intellectual property ("Background IP"). Each grants limited license to use Background IP solely for the Project.

3. JOINT CONFIDENTIAL INFORMATION
Includes:
   - Research results and experimental data
   - Technical discoveries
   - Project plans and methodologies
   - Draft publications
   - Funding information

4. PROTECTION OBLIGATIONS
Each Party agrees to:
   a) Protect Joint Confidential Information with reasonable care
   b) Share only with researchers directly involved
   c) Not publish without mutual consent
   d) Maintain accurate records
   e) Mark sensitive documents as "Confidential"

5. PUBLICATION RIGHTS
Neither Party shall publish without 30 days' notice and opportunity to review.

6. OWNERSHIP OF RESULTS
Joint inventions shall be jointly owned.

7. TERM
Continues during the Project and for [NUMBER] years thereafter."""},
        
        {"instruction": "Create an NDA for protecting trade secrets during sales discussions.",
         "output": """SALES DISCUSSION CONFIDENTIALITY AGREEMENT

This Agreement is entered into between [VENDOR] and [PROSPECT] as of [DATE].

WHEREAS, Vendor desires to provide product information to Prospect for evaluation;

NOW, THEREFORE:

1. CONFIDENTIAL INFORMATION
Includes:
   - Product demonstrations and specifications
   - Pricing and commercial terms
   - Customer references and case studies
   - Implementation methodologies
   - Roadmap and future plans
   - Competitive positioning

2. LIMITED PURPOSE
Prospect shall use information solely to evaluate purchase.

3. PROTECTION REQUIREMENTS
Prospect agrees to:
   a) Maintain confidentiality using reasonable precautions
   b) Limit disclosure to employees involved in evaluation
   c) Not share with Vendor's competitors
   d) Not use to develop competing products

4. EXCEPTIONS
Obligations do not apply to public information or independent development.

5. DURATION
Obligations continue for [NUMBER] years; trade secrets protected indefinitely.

6. NO COMMITMENT
This Agreement does not obligate either party to proceed.

7. RETURN OF MATERIALS
Upon Vendor's request, Prospect shall return all materials."""},
        
        {"instruction": "Write an NDA for freelancers and independent contractors.",
         "output": """INDEPENDENT CONTRACTOR CONFIDENTIALITY AGREEMENT

This Agreement is between [COMPANY] and [CONTRACTOR] as of [DATE].

1. ENGAGEMENT
Contractor will provide services as described in the Statement of Work.

2. CONFIDENTIAL INFORMATION
Includes all non-public information disclosed, including:
   - Client lists and contacts
   - Project specifications
   - Source code and designs
   - Business strategies
   - Work product before release

3. CONTRACTOR OBLIGATIONS
Contractor shall:
   a) Hold information in strict confidence
   b) Not disclose without written consent
   c) Use only for performing services
   d) Take reasonable precautions
   e) Not make copies except as necessary

4. SECURITY MEASURES
Contractor shall maintain:
   - Password protection on devices
   - Secure storage of documents
   - Encryption when feasible
   - Immediate notification of incidents

5. WORK PRODUCT
All deliverables are Company's property. Contractor assigns all intellectual property rights.

6. RETURN OF MATERIALS
Upon completion, Contractor shall return all materials and delete electronic copies.

7. SURVIVAL
Confidentiality obligations survive for [NUMBER] years after engagement."""},
        
        {"instruction": "Draft a confidentiality agreement for board members and advisors.",
         "output": """BOARD MEMBER/ADVISOR CONFIDENTIALITY AGREEMENT

This Agreement is between [COMPANY] and [NAME] ("Board Member/Advisor") effective [DATE].

1. ROLE
Board Member/Advisor shall serve on the [Board of Directors/Advisory Board] and will access highly sensitive information.

2. CONFIDENTIAL INFORMATION
Includes:
   - Board materials and minutes
   - Financial statements and projections
   - Strategic plans and competitive analysis
   - M&A discussions and investments
   - Personnel matters and compensation
   - Pending litigation
   - Material non-public information (MNPI)
   - Trade secrets and proprietary technology

3. FIDUCIARY OBLIGATIONS
Board Member acknowledges fiduciary duties of care, loyalty, and confidentiality.

4. NON-DISCLOSURE
Board Member/Advisor agrees to:
   a) Maintain strict confidentiality
   b) Not disclose to any third party
   c) Not use for personal benefit
   d) Not trade securities based on MNPI

5. CONFLICTS OF INTEREST
Shall promptly disclose potential conflicts and recuse as appropriate.

6. DOCUMENT HANDLING
All board materials shall be returned upon resignation or removal.

7. DURATION
Obligations continue during service and for [NUMBER] years thereafter."""},
    ]
    
    # -------------------------------------------------------------------------
    # 3. PRIVACY POLICIES (10 variations)
    # -------------------------------------------------------------------------
    privacy_templates = [
        {"instruction": "Write a privacy policy introduction for an e-commerce website.",
         "output": """PRIVACY POLICY

Last Updated: [DATE]

Welcome to [COMPANY NAME]. This Privacy Policy describes how we collect, use, disclose, and protect your personal information when you visit our website [URL], use our mobile application, or make purchases (collectively, the "Services").

By using our Services, you agree to this Privacy Policy. If you do not agree, please do not use our Services.

We may update this Privacy Policy from time to time by posting updates on this page and updating the "Last Updated" date. We encourage periodic review.

If you have questions, contact us at:
[COMPANY NAME]
[ADDRESS]
Email: privacy@[DOMAIN].com
Phone: [PHONE NUMBER]"""},
        
        {"instruction": "Draft the data collection section of a privacy policy for a mobile app.",
         "output": """INFORMATION WE COLLECT

We collect several types of information:

1. Information You Provide
   - Account information: name, email, password, phone
   - Profile information: photo, preferences, settings
   - Payment information: credit card, billing address
   - Communications: messages, feedback, support requests
   - User content: photos, posts, uploads

2. Information Collected Automatically
   - Device information: type, OS, unique identifiers
   - Usage data: features used, time spent, actions taken
   - Location data: GPS (with permission), IP-based location
   - Log data: access times, error logs, referring URLs

3. Information from Third Parties
   - Social media accounts you connect
   - Analytics providers: aggregated usage
   - Advertising partners: ad interactions

4. Cookies and Tracking
   We use cookies and similar technologies to:
   - Remember preferences
   - Authenticate accounts
   - Analyze usage patterns
   - Deliver personalized advertising

5. Sensitive Information
   We do not intentionally collect sensitive information without explicit consent."""},
        
        {"instruction": "Write the data sharing section of a privacy policy.",
         "output": """HOW WE SHARE YOUR INFORMATION

We may share your personal information as follows:

1. Service Providers
   Third-party vendors performing services including:
   - Payment processing
   - Cloud hosting
   - Email delivery
   - Customer support
   - Analytics
   
   These providers are contractually bound to protect your information.

2. Business Partners
   With your consent, we may share with partners offering relevant products.

3. Legal Requirements
   We may disclose when required to:
   - Comply with laws or legal process
   - Respond to lawful government requests
   - Protect our rights, privacy, or property
   - Enforce our terms

4. Business Transfers
   In merger, acquisition, or asset sale, your information may be transferred. We will notify you of any change in ownership.

5. Aggregated Data
   We may share aggregated or de-identified information for research or marketing.

6. Your Direction
   We may share at your direction or with explicit consent.

WE DO NOT SELL YOUR PERSONAL INFORMATION."""},
        
        {"instruction": "Create the user rights section of a GDPR-compliant privacy policy.",
         "output": """YOUR PRIVACY RIGHTS

Depending on your location, you may have these rights:

1. Right to Access
   Request a copy of your personal information and processing details.

2. Right to Rectification
   Request correction of inaccurate or incomplete information.

3. Right to Erasure
   Request deletion in certain circumstances.

4. Right to Restrict Processing
   Request limitation while verifying accuracy or assessing objections.

5. Right to Data Portability
   Request your information in a structured, machine-readable format.

6. Right to Object
   Object to processing based on legitimate interests or for marketing.

7. Automated Decision-Making Rights
   Right not to be subject to solely automated decisions significantly affecting you.

8. Right to Withdraw Consent
   Where based on consent, withdraw at any time without affecting prior processing.

HOW TO EXERCISE RIGHTS
Submit requests to privacy@[DOMAIN].com. We respond within 30 days and may require identity verification.

APPEAL PROCESS
If unsatisfied, you may lodge a complaint with your local data protection authority."""},
        
        {"instruction": "Write a data security section for a financial services privacy policy.",
         "output": """DATA SECURITY

We implement comprehensive security measures:

1. Technical Safeguards
   - 256-bit SSL/TLS encryption for transmission
   - AES-256 encryption for data at rest
   - Multi-factor authentication
   - Regular penetration testing
   - Intrusion detection systems
   - Secure data centers with 24/7 monitoring

2. Administrative Safeguards
   - Background checks for employees with data access
   - Mandatory security training
   - Access controls based on need-to-know
   - Incident response plans
   - Regular independent audits

3. Physical Safeguards
   - Restricted facility access
   - Video surveillance
   - Environmental controls

4. Regulatory Compliance
   - Gramm-Leach-Bliley Act (GLBA)
   - PCI-DSS for payment card data
   - SOC 2 Type II certification
   - State data breach notification laws

5. Data Breach Procedures
   In case of breach, we will:
   - Investigate and contain immediately
   - Notify affected individuals as required
   - Report to regulatory authorities
   - Prevent future incidents

IMPORTANT: No security system is impenetrable. We cannot guarantee absolute security."""},
        
        {"instruction": "Draft a children's privacy section (COPPA compliance).",
         "output": """CHILDREN'S PRIVACY

Our commitment to protecting children's privacy is paramount.

1. Age Restrictions
   Our Services are not intended for children under 13. We do not knowingly collect information from children under 13.

2. COPPA Compliance
   We comply with COPPA. We do not:
   - Knowingly collect from children under 13 without parental consent
   - Condition participation on excessive disclosure
   - Share children's information for marketing

3. Parental Consent
   If Services are directed to children under 13, we will:
   - Provide direct notice to parents
   - Obtain verifiable parental consent
   - Give parents opportunity to review and delete

4. Information We May Collect (with consent)
   - Username (not real name)
   - Parent's email for contact
   - Persistent identifiers for service operation only

5. Parental Rights
   Parents may:
   - Review collected information
   - Request deletion
   - Refuse further collection
   - Agree to collection but not third-party disclosure

6. Verification
   If we discover collection from a child under 13 without proper consent, we will delete immediately.

7. Contact
   Email: privacy@[DOMAIN].com"""},
        
        {"instruction": "Write a cookie policy section for a news website.",
         "output": """COOKIE POLICY

1. What Are Cookies?
   Small text files placed on your device to remember preferences and understand usage.

2. Types We Use

   Essential Cookies (Always Active)
   - Required for website functionality
   - Enable secure login
   - Cannot be disabled

   Performance Cookies
   - Collect anonymous usage statistics
   - Help understand popular content
   - Powered by Google Analytics

   Functionality Cookies
   - Remember preferences
   - Store reading history
   - Enable personalized recommendations

   Advertising Cookies
   - Deliver relevant advertisements
   - Track ad performance
   - May be set by advertising partners

3. Third-Party Cookies
   Our partners may set cookies, including:
   - Google (Analytics, AdSense)
   - Facebook Pixel
   - Twitter Analytics

4. Managing Preferences
   Control cookies through:
   - Our Cookie Preferences Center
   - Browser settings
   - Industry opt-out tools (NAI, DAA)

5. Do Not Track
   We honor Do Not Track signals where required.

6. Disabling Consequences
   Disabling may affect functionality, including:
   - Unable to save preferences
   - Less relevant content
   - Some features may not work"""},
        
        {"instruction": "Create a California Consumer Privacy Act (CCPA) compliance section.",
         "output": """CALIFORNIA PRIVACY RIGHTS (CCPA/CPRA)

California residents have additional rights:

1. Right to Know
   Request disclosure of:
   - Categories of information collected
   - Specific pieces collected
   - Sources of information
   - Purposes for collection
   - Third parties we share with

2. Right to Delete
   Request deletion, subject to exceptions.

3. Right to Correct
   Request correction of inaccurate information.

4. Right to Opt-Out
   Opt-out of:
   - Sale of personal information
   - Cross-context behavioral advertising
   - Use of sensitive information beyond necessary purposes

5. Right to Non-Discrimination
   We will not discriminate for exercising rights.

6. Categories Collected (past 12 months)
   - Identifiers (name, email, IP)
   - Commercial information (purchase history)
   - Internet activity (browsing, interactions)
   - Geolocation data
   - Professional information

7. How to Submit Requests
   - Online: [URL]
   - Email: ccpa@[DOMAIN].com
   - Phone: [TOLL-FREE NUMBER]

8. Verification
   We verify identity before fulfilling requests.

9. Authorized Agents
   You may designate an authorized agent with written permission."""},
        
        {"instruction": "Write an international data transfer section.",
         "output": """INTERNATIONAL DATA TRANSFERS

Your information may be transferred to countries other than your residence.

1. Transfer Mechanisms
   When transferring outside the EEA, UK, or Switzerland, we ensure safeguards through:

   a) Standard Contractual Clauses (SCCs)
      EU Commission-approved clauses for transfers to countries without adequacy decisions.

   b) Adequacy Decisions
      Transfers to countries with adequate protection (e.g., Canada, Japan, UK).

   c) Binding Corporate Rules
      For intra-group transfers with supervisory authority approval.

   d) Consent
      In some cases, explicit consent for specific transfers.

2. Data Storage Locations
   Primary data centers in:
   - United States
   - European Union
   
3. Additional Protections
   We apply consistent security and privacy protections regardless of location.

4. Your Rights
   You may:
   - Request information about transfers
   - Obtain a copy of safeguards
   - Object in certain circumstances

5. Contact
   For questions: dpo@[DOMAIN].com

6. EU-U.S. Data Privacy Framework
   We are [certified under/working toward certification under] the EU-U.S. Data Privacy Framework."""},
        
        {"instruction": "Draft a data retention section.",
         "output": """DATA RETENTION

We retain information only as long as necessary.

1. Retention Periods

   Account Information
   - Active accounts: While active
   - Closed accounts: Deleted within 90 days, except as required by law

   Transaction Records
   - 7 years for tax and accounting compliance
   - Payment card data deleted after processing

   Communication Records
   - Customer service: 3 years
   - Marketing consent: Duration plus 3 years

   Usage Data
   - Aggregated analytics: Indefinitely
   - Individual logs: 2 years

   Legal Hold
   - Information in legal proceedings retained until resolution

2. Factors Affecting Retention
   - Nature and sensitivity
   - Processing purposes
   - Legal requirements
   - Risk of harm from unauthorized use

3. Secure Disposal
   When periods expire, we securely delete or anonymize using:
   - Secure electronic deletion
   - Cross-cut shredding for physical documents
   - Backup purging within normal rotation

4. Your Rights
   You may request deletion, subject to legal requirements.

5. Questions
   Contact: privacy@[DOMAIN].com"""},
    ]
    
    # -------------------------------------------------------------------------
    # 4. RECOMMENDATION LETTERS (10 variations)
    # -------------------------------------------------------------------------
    recommendation_templates = [
        {"instruction": "Write a recommendation letter for a software engineer applying to graduate school.",
         "output": """[YOUR NAME]
[YOUR TITLE]
[COMPANY NAME]
[DATE]

Graduate Admissions Committee
[UNIVERSITY NAME]
[DEPARTMENT]

Dear Admissions Committee,

I enthusiastically recommend [CANDIDATE NAME] for your Master's program in Computer Science. As [CANDIDATE]'s supervisor for three years, I've observed their exceptional technical abilities and intellectual curiosity.

[CANDIDATE] distinguished themselves through outstanding problem-solving skills. They led our real-time data processing pipeline, reducing latency by 60%—a project requiring independent research and novel solutions essential for graduate study.

What sets [CANDIDATE] apart is their combination of technical expertise and communication abilities. They present complex concepts to non-technical stakeholders and mentor junior engineers effectively.

Their intellectual curiosity extends beyond job requirements. They independently explored machine learning applications and proposed a research project we're now pursuing—demonstrating the self-directed learning aptitude essential for graduate school.

I give [CANDIDATE] my highest recommendation. They rank in the top 5% of engineers I've worked with in my 15-year career.

Sincerely,
[YOUR NAME]
[YOUR TITLE]"""},
        
        {"instruction": "Draft a recommendation letter for an employee seeking a management position.",
         "output": """[YOUR NAME]
[YOUR TITLE]
[COMPANY NAME]
[DATE]

To Whom It May Concern,

I recommend [CANDIDATE NAME] for a management position. I've worked with [CANDIDATE] for five years as their department director and can speak to their exceptional leadership potential.

Leadership by Example
[CANDIDATE] demonstrates qualities essential for effective management. During our restructuring, they voluntarily coordinated cross-functional teams, showing natural leadership abilities.

People Development
[CANDIDATE]'s commitment to developing others impressed me most. They've mentored four junior team members, all since promoted—a testament to their mentorship abilities.

Results-Oriented
[CANDIDATE] consistently delivered:
- 35% efficiency improvement through process initiatives
- $500K project budget managed with zero overruns
- Customer satisfaction scores 20% above average

Problem-Solving
[CANDIDATE] approaches challenges with calm analysis and creative solutions, communicating effectively across all organizational levels.

I believe [CANDIDATE] is ready for management responsibility and will excel.

Best regards,
[YOUR NAME]
[YOUR TITLE]"""},
        
        {"instruction": "Write a recommendation letter for a scholarship application.",
         "output": """[INSTITUTION LETTERHEAD]
[DATE]

Scholarship Selection Committee
[SCHOLARSHIP NAME]
[ORGANIZATION]

Re: Scholarship Recommendation for [STUDENT NAME]

Dear Selection Committee,

It is my honor to recommend [STUDENT NAME] for the [SCHOLARSHIP NAME]. As their professor and academic advisor for two years, I've witnessed their outstanding achievement, leadership, and community service.

Academic Excellence
[STUDENT] maintains a [GPA] GPA in a challenging double major. Their research paper on [TOPIC] was selected for the undergraduate symposium and is being developed for publication.

Leadership and Service
As President of [ORGANIZATION], they increased membership by 40% and organized outreach serving 200+ students—all while working 15 hours weekly to finance their education.

Financial Need
Despite financial obstacles, [STUDENT] has never let challenges diminish their performance. This scholarship would enable greater focus on studies and research.

Character
[STUDENT] possesses maturity, integrity, and genuine kindness that distinguish them among peers.

[STUDENT] embodies this scholarship's values. I recommend them without hesitation.

Respectfully,
[YOUR NAME]
[YOUR TITLE]"""},
        
        {"instruction": "Create a recommendation letter for a medical school applicant.",
         "output": """[LETTERHEAD]
[DATE]

Office of Admissions
[MEDICAL SCHOOL NAME]

Dear Admissions Committee,

I strongly recommend [CANDIDATE NAME] for medical school. As attending physician at [HOSPITAL] where [CANDIDATE] completed a two-year clinical research fellowship, I've evaluated their potential as a future physician.

Clinical Aptitude
[CANDIDATE] demonstrated exceptional clinical acumen and patient rapport. They assisted with over 200 consultations, earning praise for their compassionate bedside manner and ability to explain complex information accessibly.

Research Contributions
[CANDIDATE] co-investigated our study on [TOPIC], contributing to data analysis and publication in [JOURNAL]. Their intellectual curiosity and methodical approach will serve them well in evidence-based practice.

Interprofessional Collaboration
[CANDIDATE] excels collaboratively, working effectively with nurses, technicians, and physicians across specialties with respect and professionalism.

Ethical Reasoning
[CANDIDATE] demonstrated sophisticated ethical reasoning during case discussions, considering multiple perspectives while maintaining commitment to patient welfare.

Personal Qualities
Beyond skills, [CANDIDATE] brings warmth, humility, and resilience. I've mentored dozens of pre-medical students and can confidently say [CANDIDATE] ranks among the top.

Sincerely,
[YOUR NAME], MD"""},
        
        {"instruction": "Write a character reference letter for a court proceeding.",
         "output": """[YOUR NAME]
[YOUR ADDRESS]
[DATE]

The Honorable [JUDGE NAME]
[COURT NAME]
[ADDRESS]

Re: Character Reference for [DEFENDANT NAME]
Case Number: [CASE NUMBER]

Your Honor,

I provide this character reference for [DEFENDANT NAME], whom I've known for [NUMBER] years as their [RELATIONSHIP].

My Background
I am a [OCCUPATION] and have lived in this community for [NUMBER] years. I have no criminal record and am willing to testify.

My Knowledge
I have regular contact with [DEFENDANT] through [CONTEXT]. I have observed them to be a person of good character with honesty and integrity.

Character Observations
[DEFENDANT] has demonstrated:
- Responsibility and reliability
- Respect for others and community standards
- Strong work ethic and family dedication
- Genuine remorse for actions bringing them before this court

Community Contributions
[DEFENDANT] has contributed positively through [EXAMPLES].

Personal Commitment
I will support [DEFENDANT] in rehabilitation efforts. I believe they have learned from this experience.

I respectfully ask the Court to consider [DEFENDANT]'s character and rehabilitation potential.

Respectfully submitted,
[YOUR SIGNATURE]
[YOUR NAME]

I declare under penalty of perjury that the foregoing is true and correct."""},
        
        {"instruction": "Draft a professional reference letter for a consulting position.",
         "output": """[YOUR NAME]
[YOUR TITLE]
[COMPANY NAME]
[DATE]

Hiring Manager
[CONSULTING FIRM]

Dear Hiring Manager,

I recommend [CANDIDATE NAME] for a consulting position. Having worked with them for four years, including two as their project manager, I attest to their exceptional analytical abilities and client relationship skills.

Analytical Excellence
[CANDIDATE] possesses outstanding analytical skills. They led financial analysis for our $50M merger evaluation, identifying $8M in synergies our initial estimates missed.

Client Management
[CANDIDATE] builds trusted client relationships, understanding needs, managing expectations, and delivering solutions that exceed requirements.

Communication Skills
[CANDIDATE] communicates complex information with clarity. Their executive presentations have been described as "best in class."

Team Leadership
As project lead, [CANDIDATE] motivates teams, delegates effectively, and creates an environment where people do their best work.

Work Ethic
[CANDIDATE] delivers under pressure, managing multiple workstreams with composure, always meeting deadlines without compromising quality.

I give my highest recommendation for a consulting career.

Sincerely,
[YOUR NAME]
[YOUR TITLE]"""},
        
        {"instruction": "Write a recommendation letter for a teaching position.",
         "output": """[SCHOOL LETTERHEAD]
[DATE]

Search Committee
[SCHOOL NAME]

Dear Search Committee,

I recommend [CANDIDATE NAME] for a teaching position. As Principal where [CANDIDATE] has taught for four years, I've directly observed their exceptional teaching abilities.

Instructional Excellence
[CANDIDATE] combines deep content knowledge with innovative pedagogy. Their students exceed state averages by 15% on assessments. More importantly, they create an environment where students develop genuine love of learning.

Differentiated Instruction
[CANDIDATE] excels at meeting diverse needs, implementing differentiation strategies that challenge advanced learners while supporting struggling students.

Student Relationships
[CANDIDATE] builds meaningful connections while maintaining boundaries. Students feel safe, respected, and motivated. Alumni frequently return to express thanks.

Professional Contributions
Beyond the classroom, [CANDIDATE] has:
- Developed new curriculum
- Mentored new teachers
- Led professional development
- Coordinated extracurricular activities

Collaboration
[CANDIDATE] collaborates effectively with teachers, administrators, and parents, communicating proactively and responding constructively.

I will miss [CANDIDATE] should they accept elsewhere. They have my unqualified endorsement.

Respectfully,
[YOUR NAME]
Principal"""},
        
        {"instruction": "Create a reference letter for a tenant application.",
         "output": """[YOUR NAME]
[YOUR ADDRESS]
[DATE]

[PROPERTY MANAGER]
[PROPERTY MANAGEMENT COMPANY]

Re: Tenant Reference for [APPLICANT NAME]

Dear [RECIPIENT NAME],

I provide a reference for [APPLICANT NAME], who was my tenant at [ADDRESS] from [START DATE] to [END DATE].

Rental History
During [APPLICANT]'s [NUMBER]-year tenancy, they demonstrated ideal tenant qualities:

Payment History
[APPLICANT] paid rent on time every month without exception. Not a single late payment in [NUMBER] years.

Property Care
[APPLICANT] maintained the property excellently. Upon move-out, only standard cleaning was needed with no repairs beyond normal wear. Full security deposit returned.

Behavior and Conduct
[APPLICANT] was respectful of neighbors and community rules. I never received complaints about noise or disturbances. They communicated easily and responded promptly.

Lease Compliance
[APPLICANT] adhered to all lease terms, including occupancy limits, pet policies, and parking regulations.

Reason for Moving
[APPLICANT] is relocating due to [REASON]. I would gladly rent to them again.

I recommend [APPLICANT] without reservation.

Sincerely,
[YOUR NAME]
Former Landlord"""},
        
        {"instruction": "Write a LinkedIn recommendation for a marketing professional.",
         "output": """I worked with [NAME] for three years at [COMPANY], where they served as Senior Marketing Manager. They are simply one of the most talented marketers I've encountered in my 15-year career.

[NAME]'s strategic thinking sets them apart. They developed our go-to-market strategy for [PRODUCT], which became our most successful launch, driving $10M in first-year revenue.

What I value most is their combination of creativity and analytical rigor. They're equally comfortable brainstorming bold campaigns and diving into performance data. Under their leadership, marketing ROI improved by 45%.

Beyond technical skills, [NAME] excels at building relationships across the organization, aligning marketing with broader company objectives.

[NAME] also invested in developing their team. Three direct reports have been promoted—a testament to their mentorship.

Any organization would be fortunate to have [NAME]. I recommend them wholeheartedly and would jump at the opportunity to work together again."""},
        
        {"instruction": "Draft a recommendation letter for a research grant application.",
         "output": """[UNIVERSITY LETTERHEAD]
[DATE]

[GRANT PROGRAM NAME]
[FUNDING ORGANIZATION]

Re: Research Grant Recommendation for Dr. [CANDIDATE NAME]

Dear Grant Review Committee,

I strongly endorse Dr. [CANDIDATE NAME]'s application for the [GRANT NAME]. As Chair of [DEPARTMENT] and collaborator on related research, I have deep familiarity with their work.

Research Excellence
Dr. [CANDIDATE] is an emerging leader in [FIELD]. Their work has produced significant findings, including [KEY DISCOVERY], published in [JOURNAL] with over [NUMBER] citations.

Proposed Project Merit
The proposed research addresses a critical gap. The approach is innovative and methodologically sound, with preliminary data demonstrating feasibility and transformative potential.

Research Environment
Dr. [CANDIDATE] will have access to exceptional resources, including our [FACILITY] and collaboration with leading researchers.

Mentorship
This grant will support development as an independent investigator. They've demonstrated leadership through mentoring students and leading research teams.

Capacity to Complete
Dr. [CANDIDATE] has a strong track record delivering projects on time and budget, with [NUMBER] publications and [NUMBER] trained students.

I believe this research will make significant contributions, and Dr. [CANDIDATE] is ideally positioned to conduct it.

Sincerely,
[YOUR NAME]
[YOUR TITLE]"""},
    ]
    
    # -------------------------------------------------------------------------
    # 5. BUSINESS EMAILS (12 variations)
    # -------------------------------------------------------------------------
    email_templates = [
        {"instruction": "Write a professional email introducing yourself to a new client.",
         "output": """Subject: Introduction - Your Dedicated Account Manager at [COMPANY]

Dear [CLIENT NAME],

I'm reaching out to introduce myself as your new Account Manager at [COMPANY]. I'll be your primary point of contact and am committed to ensuring excellent service.

About me: I've been with [COMPANY] for [NUMBER] years, specializing in [AREA]. I'm passionate about building strong client relationships and delivering results.

I've reviewed your account and am excited about supporting your goals for [PROJECT]. I'd love to schedule a brief call to introduce myself properly and learn about your priorities.

Would you have 30 minutes this week? Please suggest times that work, or book directly: [CALENDAR LINK].

Please reach out with any questions. You can reach me at [PHONE] or this email.

Best regards,
[YOUR NAME]
Account Manager"""},
        
        {"instruction": "Draft a follow-up email after a sales meeting.",
         "output": """Subject: Follow-Up: Next Steps from Our Discussion

Dear [NAME],

Thank you for meeting today. I enjoyed learning about [COMPANY]'s goals for [INITIATIVE].

Discussion Summary:
- Primary challenge: [CHALLENGE]
- Goals: [GOALS]
- Timeline: [TIMEFRAME]

Based on our conversation, [SOLUTION] would help you [BENEFIT].

Next Steps:
1. I'll send the proposal by [DATE]
2. I'll connect you with our implementation team
3. Follow-up call on [DATE] to review

Action Requested:
Could you share the technical requirements document? This will help me address your specific needs.

Attached: [MATERIALS]

Looking forward to continuing our conversation.

Best regards,
[YOUR NAME]"""},
        
        {"instruction": "Write an email requesting a meeting with a potential investor.",
         "output": """Subject: [COMPANY NAME] - Investment Opportunity in [INDUSTRY]

Dear [INVESTOR NAME],

I'm [YOUR NAME], Founder of [COMPANY]. [MUTUAL CONNECTION] suggested you might be interested given your focus on [SECTOR].

[COMPANY] is [ONE-SENTENCE DESCRIPTION]. We've achieved [KEY METRIC] and are raising [ROUND SIZE] to [USE OF FUNDS].

Why now:
- Market: [MARKET SIZE]
- Traction: [KEY METRICS]
- Team: [CREDENTIALS]
- Differentiation: [ADVANTAGE]

We've secured commitments from [NOTABLE INVESTORS] and are looking for partners who bring [SPECIFIC VALUE].

Would you have 30 minutes for a call? I'm happy to work around your schedule.

Executive summary attached. Full deck available upon request.

Best regards,
[YOUR NAME]
Founder & CEO"""},
        
        {"instruction": "Create an email announcing an organizational change.",
         "output": """Subject: Important Announcement: Leadership Update

Dear Team,

I want to share an important update before external announcements.

Effective [DATE], [NAME] will become [NEW TITLE], reporting to [STRUCTURE].

This change reflects [RATIONALE]. [NAME] brings [QUALIFICATIONS] and will [EXPECTED IMPACT].

What this means:
- [DEPARTMENT] will report to [NAME]
- Day-to-day operations continue unchanged

What's next:
- [NAME] will hold a town hall on [DATE]
- Your manager will discuss specific impacts
- Questions to [CONTACT]

Please join me in congratulating [NAME].

Best regards,
[YOUR NAME]"""},
        
        {"instruction": "Write an email apologizing for a service failure.",
         "output": """Subject: Our Apology and Resolution for [ISSUE]

Dear [CUSTOMER NAME],

I personally apologize for [ISSUE] on [DATE]. This fell short of our standards, and I understand the frustration caused.

What happened:
[EXPLANATION]

What we've done:
- [CORRECTIVE ACTION]
- [COMPENSATION PROVIDED]
- [LONG-TERM FIX]

To make this right:
We've applied [COMPENSATION] to your account, reflected in [TIMEFRAME].

Preventing recurrence:
We've implemented [IMPROVEMENTS] and are committed to continuous improvement.

I value your business and hope for another opportunity. Please contact me directly at [PHONE/EMAIL] with concerns.

Sincerely,
[YOUR NAME]
[TITLE]"""},
        
        {"instruction": "Draft an email declining a job offer professionally.",
         "output": """Subject: Thank You - [POSITION] Offer

Dear [HIRING MANAGER NAME],

Thank you for the [POSITION] offer at [COMPANY]. I appreciate the time you and the team invested.

After careful consideration, I've decided to decline. This wasn't easy—I was genuinely impressed by [POSITIVE ASPECT]. However, after reflecting on my career goals, I believe this is the right choice.

This decision reflects no issue with [COMPANY]. [BRIEF REASON IF APPROPRIATE].

I have great respect for what [COMPANY] is building and hope our paths cross again. Please extend my thanks to everyone I met.

With gratitude,
[YOUR NAME]"""},
        
        {"instruction": "Write an email negotiating contract terms with a vendor.",
         "output": """Subject: Proposal Review - [CONTRACT NAME]

Dear [VENDOR NAME],

Thank you for the proposal. We've reviewed it and are excited about the partnership. Before proceeding, I'd like to discuss a few terms.

Points for discussion:

1. Pricing
   The proposed [AMOUNT] exceeds our budget. We propose [COUNTER-OFFER] based on market research.

2. Payment Terms
   We typically operate Net-60 rather than Net-30.

3. Contract Duration
   We'd prefer [COUNTER TERM] initially with mutual renewal options.

4. SLA
   We'd like specific uptime guarantees and response time commitments with service credits.

5. Termination
   We'd like to modify notice from [PROPOSED] to [COUNTER] days.

We're committed to moving forward. Could we schedule a call this week?

Best regards,
[YOUR NAME]"""},
        
        {"instruction": "Create an email requesting a deadline extension.",
         "output": """Subject: Request: Timeline Adjustment for [PROJECT NAME]

Dear [CLIENT NAME],

I'm writing regarding [PROJECT] scheduled for [ORIGINAL DATE].

After thorough review, delivering quality work meeting our standards by [ORIGINAL DATE] will be challenging given [REASON].

What happened:
[EXPLANATION]

Our request:
A [NUMBER]-day extension to [NEW DATE]. This will allow us to:
- [IMPROVEMENT 1]
- [IMPROVEMENT 2]
- [QUALITY ASSURANCE]

What we've done:
- [PROGRESS TO DATE]
- [MITIGATING ACTIONS]

Our commitment:
We propose [GOODWILL GESTURE] to demonstrate our commitment.

I understand this may impact planning and apologize for inconvenience. I'm available to discuss further.

Best regards,
[YOUR NAME]"""},
        
        {"instruction": "Write a networking email to a former colleague.",
         "output": """Subject: Catching Up + Quick Question

Hi [NAME],

It's been [TIME] since [COMPANY]! I've followed your journey to [CURRENT ROLE] with interest. Congratulations on [ACHIEVEMENT].

Two reasons for reaching out:

First, I'd love to catch up. I have fond memories of [PROJECT] and always valued your perspective.

Second, I wanted to ask advice. I'm currently [YOUR SITUATION] and thought you might have insights given your experience with [EXPERTISE].

I'm not looking for anything major—just a 15-20 minute call if you have time. No pressure if your schedule doesn't allow.

If open to it, let me know times that work, or grab time on my calendar: [LINK].

Best regards,
[YOUR NAME]"""},
        
        {"instruction": "Draft an email to resign from a position professionally.",
         "output": """Subject: Resignation - [YOUR NAME]

Dear [MANAGER NAME],

I formally notify you of my resignation as [TITLE] at [COMPANY], effective [LAST DAY].

This decision wasn't made lightly. My time here has been rewarding, and I'm grateful for opportunities to [EXPERIENCES]. Working with you on [PROJECT] has been a career highlight.

I've accepted a position that [BRIEF REASON]. While excited about this new chapter, I leave with deep appreciation.

During my remaining time, I commit to ensuring smooth transition:
- Complete [CURRENT PROJECTS]
- Document processes
- Train colleagues
- Support transition as needed

Please let me know how I can help make this seamless.

I'd welcome staying in touch and wish you and the team continued success.

Sincerely,
[YOUR NAME]"""},
        
        {"instruction": "Write an email requesting feedback after project completion.",
         "output": """Subject: Request for Feedback - [PROJECT NAME]

Dear [CLIENT NAME],

Now that [PROJECT] is completed, I'd like to request your feedback. Your perspective is invaluable as we continuously improve.

Please share thoughts on:

1. Project Outcomes
   - Did deliverables meet expectations?
   - Were objectives achieved?

2. Process and Communication
   - How was our responsiveness?
   - Was the timeline realistic?

3. Team Performance
   - How was your experience with our team?

4. Areas for Improvement
   - What could we do better?

5. Overall Satisfaction
   - Rating 1-10?
   - Would you recommend our services?

Reply via email, or I'd happily schedule a brief call.

If satisfied, we'd appreciate a testimonial or connection to colleagues who might benefit.

Thank you for the opportunity. I look forward to future collaborations.

Best regards,
[YOUR NAME]"""},
        
        {"instruction": "Create a cold outreach email for business development.",
         "output": """Subject: Quick Question About [COMPANY]'s [CHALLENGE]

Hi [NAME],

I noticed [OBSERVATION] and thought you might be dealing with [CHALLENGE].

Quick context: I'm [YOUR NAME] from [YOUR COMPANY]. We help [CUSTOMER TYPE] with [VALUE PROPOSITION]. Recently, we worked with [SIMILAR COMPANY] to achieve [RESULT].

I'm reaching out because [PERSONALIZED REASON].

Would you be open to a 15-minute call? I promise not to give a sales pitch—just a conversation to see if there's potential fit.

If you're not the right person, I'd appreciate guidance on who might be.

No hard feelings if timing isn't right.

Thanks,
[YOUR NAME]

P.S. [VALUE-ADD: relevant resource]"""},
    ]
    
    # -------------------------------------------------------------------------
    # 6. SALES PROPOSALS (10 variations)
    # -------------------------------------------------------------------------
    sales_templates = [
        {"instruction": "Write an executive summary for a software implementation proposal.",
         "output": """EXECUTIVE SUMMARY

Proposal: Enterprise Software Implementation
Prepared for: [CLIENT COMPANY]
Date: [DATE]

The Opportunity
[CLIENT COMPANY] faces challenges with [PAIN POINTS] costing approximately $[AMOUNT] annually.

Our Recommendation
We propose implementing [SOLUTION NAME] that will:
- Automate [PROCESS], reducing time by [X]%
- Integrate [SYSTEMS] for real-time visibility
- Scale to support [GROWTH PROJECTION]
- Ensure [REGULATION] compliance

Projected Results
Based on similar implementations:
- ROI: [X]% within [TIMEFRAME]
- Cost savings: $[AMOUNT] annually
- Productivity gains: [X] hours saved
- Error reduction: [X]% improvement

Investment Overview
Total: $[AMOUNT]
Timeline: [X] months
Payment terms: [TERMS]

Why [YOUR COMPANY]
With [X] years experience and [NUMBER] implementations, we bring proven expertise and 98% client retention.

Next Steps
Schedule a discovery workshop to finalize requirements.

Contact: [NAME], [TITLE]
[EMAIL] | [PHONE]"""},
        
        {"instruction": "Draft a pricing section for a marketing services proposal.",
         "output": """INVESTMENT AND PRICING

Option A: Essential Package
Monthly: $[AMOUNT] | Minimum: 6 months

Included:
- Social media management (3 platforms)
- Content creation (8 posts/month)
- Monthly reporting
- Dedicated account manager
- Quarterly strategy reviews

Option B: Growth Package (Recommended)
Monthly: $[AMOUNT] | Minimum: 12 months

Includes Essential, plus:
- 5 platforms, 16 posts/month
- Email marketing (4 campaigns/month)
- SEO optimization
- Paid advertising management
- Bi-weekly calls

Option C: Enterprise Package
Monthly: $[AMOUNT] | Minimum: 12 months

Includes Growth, plus:
- Unlimited platforms
- Daily content
- Video production (2/month)
- Dedicated strategy team
- Weekly executive reporting

One-Time Setup: $[AMOUNT]

Additional Services:
- Video production: $[AMOUNT]/video
- Photography: $[AMOUNT]/session

Payment Terms:
- Monthly invoicing, Net-15
- Annual prepayment: 10% discount"""},
        
        {"instruction": "Write the scope of work section for a consulting proposal.",
         "output": """SCOPE OF WORK

Project: [PROJECT NAME]
Duration: [X] weeks

1. PROJECT OVERVIEW
This engagement delivers [DELIVERABLE] through [KEY PHASES].

2. PHASE 1: DISCOVERY (Weeks 1-2)
   Activities:
   - Stakeholder interviews (up to 15 participants)
   - Process mapping workshops
   - Technology assessment
   
   Deliverables:
   - Current State Assessment
   - Requirements Document
   - Gap Analysis

3. PHASE 2: STRATEGY DEVELOPMENT (Weeks 3-4)
   Activities:
   - Future state design sessions
   - Solution architecture
   - Financial modeling
   
   Deliverables:
   - Future State Blueprint
   - Implementation Roadmap
   - Business Case

4. PHASE 3: IMPLEMENTATION PLANNING (Weeks 5-6)
   Activities:
   - Detailed project planning
   - Vendor evaluation
   - Change management planning
   
   Deliverables:
   - Implementation Plan
   - Vendor Recommendation
   - Training Plan

5. OUT OF SCOPE
   - Implementation execution
   - Software licensing
   - Third-party vendor fees

6. ASSUMPTIONS
   - Client provides timely stakeholder access
   - Relevant documentation made available"""},
        
        {"instruction": "Create a company qualifications section for a proposal.",
         "output": """COMPANY QUALIFICATIONS

About [COMPANY NAME]
Founded in [YEAR], we are a leading provider of [SERVICES] serving [CUSTOMERS] across [INDUSTRIES]. With [X] years experience, we deliver measurable results.

By the Numbers:
- [X]+ successful projects
- [X]+ satisfied clients
- [X]% retention rate
- [X] certifications
- [X]+ team members

Industry Expertise
We specialize in [INDUSTRIES] with deep domain knowledge enabling tailored solutions.

Our Approach
Our methodology combines:
- [ELEMENT 1]
- [ELEMENT 2]
- [ELEMENT 3]

Certifications and Partnerships:
- [CERTIFICATION 1]
- [CERTIFICATION 2]
- [PARTNERSHIP 1]

Awards:
- [AWARD 1]
- [AWARD 2]

Client Testimonials:
"[TESTIMONIAL]"
— [NAME], [TITLE], [COMPANY]

References Available Upon Request"""},
        
        {"instruction": "Write an implementation timeline for a project proposal.",
         "output": """IMPLEMENTATION TIMELINE

Duration: [X] Months
Target Completion: [DATE]

PHASE 1: INITIATION (Weeks 1-2)
Week 1:
☐ Kickoff meeting
☐ Team introductions
☐ Communication plan

Week 2:
☐ Requirements gathering
☐ Risk assessment
☐ Project plan finalization

Milestone: Project Charter Approved

PHASE 2: DEVELOPMENT (Weeks 3-8)
Weeks 3-4: Design and architecture
Weeks 5-6: Core development
Weeks 7-8: Feature completion, internal QA

Milestone: Development Complete

PHASE 3: TESTING (Weeks 9-10)
Week 9: UAT, performance testing
Week 10: Issue resolution, sign-off

Milestone: UAT Approved

PHASE 4: DEPLOYMENT (Weeks 11-12)
Week 11: Environment prep, training
Week 12: Go-live, hypercare begins

Milestone: Go-Live Complete

POST-IMPLEMENTATION (Weeks 13-16)
☐ 30-day hypercare
☐ Optimization
☐ Project closure

Final Milestone: Project Closure"""},
        
        {"instruction": "Draft a terms and conditions section for a service proposal.",
         "output": """TERMS AND CONDITIONS

1. ACCEPTANCE
   Proposal valid for [30/60/90] days. Acceptance confirmed upon signature.

2. FEES AND PAYMENT
   - Fees as specified in Pricing section
   - Exclusive of applicable taxes
   - Invoices issued [monthly/upon milestone]
   - Payment due NET-30
   - Late payments accrue [X]% monthly interest
   - Pre-approved expenses invoiced at cost plus [X]%

3. CHANGE ORDERS
   - Changes require written change order
   - Will specify timeline and fee impact
   - Work begins only after approval

4. CLIENT RESPONSIBILITIES
   - Timely provision of information and decisions
   - Assignment of project contact with authority
   - Stakeholder availability
   - Review of deliverables within [X] business days

5. INTELLECTUAL PROPERTY
   - Client owns deliverables upon full payment
   - [COMPANY] retains pre-existing IP

6. CONFIDENTIALITY
   - Both parties maintain confidentiality
   - Survives termination for [X] years

7. WARRANTY
   - Deliverables conform to specifications for [X] days
   - Exclusive remedy is re-performance

8. LIABILITY
   - Limited to fees paid
   - Neither party liable for indirect damages

9. TERMINATION
   - Either party may terminate with [30] days notice
   - Client pays for completed work

10. GOVERNING LAW
    Governed by laws of [STATE]"""},
        
        {"instruction": "Write a problem statement and solution overview for an IT proposal.",
         "output": """UNDERSTANDING YOUR CHALLENGES

Current State Assessment

1. Aging Infrastructure
   Your [X]-year-old servers create:
   - Increased failure risk and downtime
   - Limited vendor support
   - Rising maintenance costs (~$[X] annually)

2. Scalability Constraints
   - Processing at [X]% utilization
   - Storage at [X]% capacity
   - Manual provisioning requiring [X] weeks

3. Security Gaps
   - [X] unsupported systems
   - Inconsistent backup capabilities
   - Potential compliance exposure

4. Operational Inefficiency
   IT spending [X]% on maintenance vs. innovation

OUR PROPOSED SOLUTION

Phase 1: Foundation Modernization
Migrate to [PLATFORM] featuring:
- Hyperconverged infrastructure
- [X]x performance improvement
- Built-in redundancy
- [X]-year warranty with 24/7 support

Phase 2: Cloud Integration
- Cloud bursting for peak demand
- [X]-minute RPO, [X]-hour RTO disaster recovery
- Pay-as-you-go capacity

Phase 3: Automation
- Single-pane management
- Automated provisioning (weeks to hours)
- AI-powered monitoring
- Self-service portal

Expected Outcomes:
✓ 99.99% uptime (vs. current 99.5%)
✓ [X]% cost reduction
✓ [X]x faster deployment
✓ Full compliance"""},
        
        {"instruction": "Create a case study section demonstrating relevant experience.",
         "output": """RELEVANT EXPERIENCE

CASE STUDY 1: [INDUSTRY] TRANSFORMATION

Client: [COMPANY] (Fortune 500)
Challenge: [DESCRIPTION]
Duration: [TIMEFRAME]

Situation:
[COMPANY] struggled with:
- [Challenge 1]
- [Challenge 2]
- [Challenge 3]

Solution:
We implemented [SOLUTION] including:
- [Component 1]
- [Component 2]
- [Component 3]

Results:
✓ 40% cost reduction
✓ 3x productivity improvement
✓ 99.9% uptime
✓ $10M new product launch enabled

Client Quote:
"[TESTIMONIAL]"
— [Name], [Title]

---

CASE STUDY 2: RAPID IMPLEMENTATION

Client: [COMPANY]
Challenge: [DESCRIPTION]
Duration: [TIMEFRAME]

Situation: Needed [OBJECTIVE] within [CONSTRAINT].

Solution: Delivered through [APPROACH].

Results:
✓ [X] weeks ahead of schedule
✓ [Quantified result]
✓ [Quantified result]

---

Additional references available upon request."""},
        
        {"instruction": "Write a risk mitigation section for a project proposal.",
         "output": """RISK ASSESSMENT AND MITIGATION

RISK 1: SCOPE CREEP
Probability: Medium | Impact: High

Mitigation:
- Detailed requirements and sign-off
- Formal change control process
- Weekly scope reviews
- Clear in-scope/out-of-scope documentation

Contingency: 10% budget reserve for approved changes

---

RISK 2: RESOURCE AVAILABILITY
Probability: Medium | Impact: Medium

Mitigation:
- Backup stakeholders identified
- Meetings scheduled 2+ weeks ahead
- Asynchronous communication options

Contingency: Built-in schedule buffers

---

RISK 3: TECHNICAL INTEGRATION
Probability: Medium | Impact: High

Mitigation:
- Early technical discovery and proof-of-concept
- Specialists with [TECHNOLOGY] experience
- Phased integration with testing
- Documented rollback procedures

Contingency: Vendor support escalation path

---

RISK 4: DATA MIGRATION
Probability: Medium | Impact: High

Mitigation:
- Data profiling in discovery
- Data cleansing prior to migration
- Phased migration with validation

Contingency: Extended parallel operation

---

RISK GOVERNANCE
- Weekly risk review in status meetings
- Risk register maintained throughout
- Immediate escalation for high-probability/high-impact risks"""},
        
        {"instruction": "Draft a partnership benefits section for a strategic proposal.",
         "output": """WHY PARTNER WITH [YOUR COMPANY]

1. PROVEN EXPERTISE
   - [X]+ years focused on [SPECIALTY]
   - [X]+ implementations in [INDUSTRY]
   - [X]+ certified professionals
   - Recognized leader by [ANALYST FIRM]

2. INDUSTRY SPECIALIZATION
   Unlike generalists, we focus on [DOMAIN]:
   - Deep industry understanding
   - Pre-built solutions for common use cases
   - Leading vendor relationships
   - Team members with operational experience

3. METHODOLOGY THAT DELIVERS
   Our [METHODOLOGY] approach ensures:
   - [X]% projects on time
   - [X]% on budget
   - [X]% client satisfaction
   - Structured knowledge transfer

4. COMMITMENT TO SUCCESS
   - Dedicated account team
   - Executive sponsor for every project
   - Quarterly business reviews
   - Success metrics tracked from day one

5. FLEXIBLE ENGAGEMENT
   - Fixed-price for defined scope
   - Time-and-materials for evolving needs
   - Managed services for ongoing support

6. INNOVATION PARTNERSHIP
   As a client, gain access to:
   - Early access to new solutions
   - Client advisory board participation
   - Exclusive industry events
   - Thought leadership sharing

7. FINANCIAL STABILITY
   - [X] years in business
   - [X]% YoY growth
   - Long-term market commitment

THE [YOUR COMPANY] DIFFERENCE
You're not just buying a service—you're gaining a committed partner invested in your success."""},
    ]
    
    # -------------------------------------------------------------------------
    # 7. MEETING MINUTES (10 variations)
    # -------------------------------------------------------------------------
    meeting_templates = [
        {"instruction": "Write meeting minutes for a project kickoff meeting.",
         "output": """PROJECT KICKOFF MEETING MINUTES

Project: [PROJECT NAME]
Date: [DATE]
Time: [TIME]
Location: [LOCATION]

ATTENDEES
Present: [NAMES AND TITLES]
Absent: [NAME] - [REASON]

AGENDA ITEMS

1. PROJECT OVERVIEW
   - Primary goal: [OBJECTIVE]
   - Success criteria defined and agreed
   - Project charter reviewed and approved

2. SCOPE REVIEW
   - In-scope confirmed: [LIST]
   - Out-of-scope clarified: [LIST]
   - Change control process explained

3. TIMELINE
   - Phase 1: [DATE]
   - Phase 2: [DATE]
   - Go-live: [DATE]

4. ROLES
   - Project Sponsor: [NAME]
   - Project Manager: [NAME]
   - Technical Lead: [NAME]
   - Client POC: [NAME]

5. COMMUNICATION
   - Weekly status: [DAY], [TIME]
   - Status reports: Every Friday

DECISIONS
☑ Project charter approved
☑ Scope finalized
☑ Go-live date confirmed

ACTION ITEMS
| Action | Owner | Due |
| Distribute project plan | [NAME] | [DATE] |
| Schedule meetings | [NAME] | [DATE] |
| Complete risk assessment | [NAME] | [DATE] |

NEXT MEETING: [DATE], [TIME]

Minutes by: [NAME]"""},
        
        {"instruction": "Create meeting minutes for a board of directors meeting.",
         "output": """BOARD OF DIRECTORS MEETING MINUTES

[COMPANY NAME]
Quarterly Board Meeting
Date: [DATE]
Location: [LOCATION]

CALL TO ORDER
Called to order at [TIME] by [CHAIR NAME].

ATTENDANCE
Present: [BOARD MEMBERS]
Absent: [NAME] - excused
Also Present: [CFO], [GENERAL COUNSEL]

A quorum was established.

APPROVAL OF MINUTES
Motion: Approve [DATE] minutes
Moved: [NAME] | Seconded: [NAME]
Vote: Approved unanimously

FINANCIAL REPORT
CFO presented Q[X] results:
- Revenue: $[X]M ([X]% vs. prior year)
- Net Income: $[X]M
- Cash Position: $[X]M

Motion: Accept financial report
Vote: Approved unanimously

CEO REPORT
- Key achievements: [SUMMARY]
- Challenges: [SUMMARY]
- Strategic initiatives: [SUMMARY]

COMMITTEE REPORTS
Audit Committee: No material findings
Compensation Committee: [RECOMMENDATIONS]

NEW BUSINESS
1. Strategic Plan Approval
   Motion: Approve [YEAR] Strategic Plan
   Vote: Approved [X]-[X]-[X]

2. Capital Expenditure
   Motion: Authorize $[AMOUNT] for [PURPOSE]
   Vote: Approved unanimously

EXECUTIVE SESSION
Entered at [TIME], exited at [TIME]. No action taken.

ADJOURNMENT
Adjourned at [TIME].

NEXT MEETING: [DATE]

____________________________
[NAME], Corporate Secretary"""},
        
        {"instruction": "Write meeting minutes for a weekly team status meeting.",
         "output": """WEEKLY TEAM STATUS MEETING

Team: [TEAM NAME]
Date: [DATE]
Facilitator: [NAME]

ATTENDEES
Present: [NAMES]
Absent: [NAME] (PTO)

PREVIOUS ACTION ITEMS
| Item | Owner | Status |
| Complete documentation | [NAME] | Done ✓ |
| Fix bug #1234 | [NAME] | In Progress |
| Review report | [NAME] | Done ✓ |

STATUS UPDATES

[NAME]:
- Completed: [TASKS]
- In Progress: [TASK] (est. completion: [DATE])
- Blockers: Waiting on design review

[NAME]:
- Completed: [TASKS]
- In Progress: [TASKS]
- Blockers: None

[NAME]:
- Completed: [TASKS]
- In Progress: [TASK]
- Note: OOO [DATES]

PROJECT STATUS
Sprint Progress: [X]% complete
On Track: [YES/NO]

BLOCKERS
- [NAME] blocked on design review - escalating
- Access request pending - IT ticket #[NUMBER]

ANNOUNCEMENTS
- All-hands Thursday 2 PM
- Code freeze begins [DATE]

ACTION ITEMS
| Action | Owner | Due |
| Follow up on review | [NAME] | [DATE] |
| Submit escalation | [NAME] | Today |

NEXT MEETING: [DATE]
Notes by: [NAME]"""},
        
        {"instruction": "Draft meeting minutes for a requirements gathering session.",
         "output": """REQUIREMENTS GATHERING SESSION

Project: [PROJECT NAME]
Session: Requirements Workshop #[NUMBER]
Date: [DATE]

PARTICIPANTS
[YOUR COMPANY]: [NAMES AND ROLES]
[CLIENT]: [NAMES AND ROLES]

OBJECTIVES
1. Document requirements for [FUNCTIONAL AREA]
2. Identify pain points and improvements
3. Define acceptance criteria
4. Prioritize requirements

CURRENT STATE
[SME] described current process:
- Pain points:
  • [PAIN POINT 1] - impacts [X] users
  • [PAIN POINT 2] - [X] hours/week manual work
- Workarounds: [DESCRIPTION]

REQUIREMENTS DOCUMENTED

REQ-001: [REQUIREMENT]
Description: System shall [DESCRIPTION]
Priority: Must Have
Acceptance Criteria:
- [CRITERION 1]
- [CRITERION 2]

REQ-002: [REQUIREMENT]
Description: System shall [DESCRIPTION]
Priority: Should Have

NON-FUNCTIONAL REQUIREMENTS
- Performance: Response time <[X] seconds
- Security: [METHOD] authentication
- Integration: [SYSTEMS]

QUESTIONS
| Question | Response | Status |
| [QUESTION] | [RESPONSE] | Resolved |
| [QUESTION] | TBD | Follow-up |

PARKING LOT
- [ITEM] - address in future session

ACTION ITEMS
| Action | Owner | Due |
| Distribute requirements | [NAME] | [DATE] |
| Provide sample data | [CLIENT] | [DATE] |

NEXT SESSION: [DATE]
Topic: [NEXT AREA]

Prepared by: [NAME]"""},
        
        {"instruction": "Create meeting minutes for a sprint retrospective.",
         "output": """SPRINT RETROSPECTIVE

Sprint: [NUMBER]
Duration: [START] - [END]
Date: [DATE]
Facilitator: [NAME]

ATTENDEES: [NAMES]

SPRINT SUMMARY
- Committed: [X] story points
- Completed: [Y] points
- Velocity: [Y]
- Completion: [Z]%

WHAT WENT WELL 👍
1. Collaboration
   - "Pair programming very effective" - [NAME]
   - Great support on complex integration

2. Technical Achievements
   - Deployed [FEATURE] with zero issues
   - Test coverage increased to [X]%

3. Process Improvements
   - Standups stayed under 15 minutes
   - Clearer requirements from refinement

Top Votes: Pair programming ⭐⭐⭐⭐⭐

WHAT COULD IMPROVE 👎
1. Blockers
   - Waited 3 days for environment access
   - External dependency delays

2. Sprint Planning
   - Stories underestimated
   - Late scope addition disrupted flow

3. Technical Debt
   - Legacy code slowed development

Top Priority: External dependencies ⭐⭐⭐⭐⭐

ACTION ITEMS
| Action | Owner | Sprint |
| Escalation path for environments | [NAME] | Immediate |
| Estimation buffer for legacy | Team | Next Sprint |
| Earlier refinement | [NAME] | Next Sprint |

TEAM HEALTH (1-5)
- Teamwork: 4.2 ↑
- Codebase: 3.5 →
- Process: 4.0 ↑
- Morale: 4.5 ↑

SHOUT-OUTS 🎉
- [NAME] for debugging help
- Team for achieving goal despite challenges

Notes by: [NAME]"""},
        
        {"instruction": "Write meeting minutes for an all-hands company meeting.",
         "output": """ALL-HANDS MEETING MINUTES

[COMPANY NAME]
Quarterly All-Hands
Date: [DATE]
Attendance: [X] total

OPENING REMARKS
[CEO] welcomed employees and thanked them for dedication during [PERIOD].

COMPANY PERFORMANCE
Q[X] Results:
- Revenue: $[X]M ([X]% YoY)
- New Customers: [X]
- Retention: [X]%
- Headcount: [X]

Key Achievements:
- Launched [PRODUCT]
- Expanded into [MARKET]
- Won [AWARD]

STRATEGIC UPDATE
Focus Areas Next Quarter:
1. [PRIORITY 1]
2. [PRIORITY 2]
3. [PRIORITY 3]

DEPARTMENT HIGHLIGHTS
Engineering: [X] releases shipped
Sales: Pipeline growth, major wins
Product: Roadmap preview

EMPLOYEE RECOGNITION
- [AWARD]: [NAME] - for [ACHIEVEMENT]
- Team Recognition: [TEAM] - for [ACHIEVEMENT]

Q&A SESSION
Q: [QUESTION]
A: [RESPONSE]

Q: [QUESTION]
A: [RESPONSE]

ANNOUNCEMENTS
- [ANNOUNCEMENT 1]
- [ANNOUNCEMENT 2]

CLOSING
[CEO] thanked employees and expressed confidence in achieving goals.

NEXT ALL-HANDS: [DATE]

Resources: [LINKS]
Recording: [LINK]

Minutes by: [NAME]"""},
        
        {"instruction": "Draft meeting minutes for a vendor selection committee meeting.",
         "output": """VENDOR SELECTION COMMITTEE MINUTES

Project: [PROJECT] - Vendor Selection
Date: [DATE]

COMMITTEE MEMBERS: [NAMES AND ROLES]
All present.

PURPOSE
Final vendor evaluation and selection recommendation.

BACKGROUND
RFP issued: [DATE]
Proposals received: [NUMBER]
Demos completed: [DATE RANGE]

EVALUATION SUMMARY
| Criteria (Weight) | Vendor A | Vendor B | Vendor C |
| Functionality (30%) | 4.2 | 3.8 | 4.0 |
| Technical Fit (25%) | 4.0 | 4.3 | 3.5 |
| Cost (20%) | 3.5 | 4.2 | 4.5 |
| Vendor Stability (15%) | 4.5 | 3.8 | 3.2 |
| Weighted Total | 4.06 | 4.02 | 3.82 |

DISCUSSION
Vendor A: Best functional fit, strong references, higher cost
Vendor B: Superior technical architecture, financial stability concerns
Vendor C: Missing key requirement

REFERENCE CHECK
| Vendor | Feedback |
| Vendor A | Very Positive |
| Vendor B | Mixed |
| Vendor C | Concerns raised |

DECISION
Motion: Recommend Vendor A, subject to successful negotiation
Conditions:
- [X]% pricing reduction
- SLA terms to include [REQUIREMENTS]
- [X]-year contract

Vote: Unanimous approval

NEXT STEPS
| Action | Owner | Due |
| Notify successful vendor | [NAME] | [DATE] |
| Initiate negotiation | [NAME] | [DATE] |
| Brief executive sponsor | [NAME] | [DATE] |

CONFIDENTIALITY NOTE
Contents confidential until vendor notification complete.

Approved by: [CHAIR]"""},
        
        {"instruction": "Create meeting minutes for a crisis management meeting.",
         "output": """CRISIS MANAGEMENT MEETING MINUTES
CONFIDENTIAL

Incident: [CODE]
Date: [DATE]
Classification: [SEVERITY]

CRISIS TEAM
Present: [NAMES AND ROLES]

SITUATION
Incident: [DESCRIPTION]
Detected: [TIME]
Status: [ACTIVE/CONTAINED]

Impact:
- Customers Affected: [NUMBER]
- Revenue Impact: $[AMOUNT]
- Reputational Risk: [LEVEL]
- Regulatory: [YES/NO]

TIMELINE
| Time | Event |
| [TIME] | Initial detection |
| [TIME] | First response |
| [TIME] | Crisis team activated |
| [TIME] | Current status |

STATUS BY FUNCTION
Technical: Root cause [identified/investigating], ETA [TIME/TBD]
Communications: Internal [STATUS], customer [STATUS]
Legal: Regulatory notification [REQUIRED/NOT]
Customer: Ticket volume [X]% increase

DECISIONS
1. [DECISION 1] - Approved by [ROLE]
2. [DECISION 2] - Approved by [ROLE]

IMMEDIATE ACTIONS
| Priority | Action | Owner | Due |
| P1 | [CRITICAL] | [NAME] | ASAP |
| P1 | [CRITICAL] | [NAME] | [TIME] |
| P2 | [HIGH] | [NAME] | [TIME] |

COMMUNICATION PLAN
| Audience | Channel | Timing | Owner |
| Employees | Email | [TIME] | [NAME] |
| Customers | Email | [TIME] | [NAME] |

NEXT CHECK-IN: [TIME]

Prepared by: [NAME]
Distribution: Crisis Team Only"""},
        
        {"instruction": "Write meeting minutes for a budget planning committee meeting.",
         "output": """BUDGET PLANNING COMMITTEE MINUTES

Fiscal Year: [FY]
Date: [DATE]

COMMITTEE: [NAMES AND ROLES]

OBJECTIVES
1. Review budget submissions
2. Identify gaps vs. targets
3. Prioritize investments

FINANCIAL CONTEXT
Current Year: Revenue $[X]M ([X]% of plan)

FY[NEXT] Assumptions:
- Revenue growth: [X]%
- EBITDA margin: [X]%
- Headcount: [X] FTEs
- Capital budget: $[X]M

BUDGET REQUESTS
| Dept | Current | Request | Change |
| Sales | $[X]M | $[X]M | +[X]% |
| Marketing | $[X]M | $[X]M | +[X]% |
| Engineering | $[X]M | $[X]M | +[X]% |
| Total | $[X]M | $[X]M | +[X]% |

Gap: $[X]M ([X]% reduction needed)

PRIORITIZATION

Must-Fund:
1. [ITEM] - $[X]K - Critical
2. [ITEM] - $[X]K - Revenue dependency

Should-Fund:
1. [ITEM] - $[X]K - ROI: [X]%

Deferred:
1. [ITEM] - Contingent on [CONDITION]

DECISIONS
1. Headcount: [X] FTEs approved
2. Capital: $[X]M approved
3. Contingency: $[X]K reserved

NEXT STEPS
| Action | Owner | Due |
| Revise budgets | All VPs | [DATE] |
| Consolidate | Finance | [DATE] |
| Board presentation | CFO | [DATE] |

Approved by: [CFO]"""},
        
        {"instruction": "Draft meeting minutes for an employee town hall Q&A.",
         "output": """EMPLOYEE TOWN HALL Q&A MINUTES

Topic: Open Q&A with Leadership
Date: [DATE]
Moderator: [NAME]

LEADERSHIP PANEL: [CEO], [COO], [CHRO], [CFO]

ATTENDANCE
- Participants: [NUMBER]
- Questions Submitted: [NUMBER]
- Questions Answered: [NUMBER]

OPENING
[CEO] thanked employees and reaffirmed commitment to transparency.

Q&A

COMPANY STRATEGY

Q: What's our biggest competitive threat?
A: [CEO] acknowledged [COMPETITOR] as key challenge. Responding through [STRATEGY] and differentiation in [AREAS].

Q: Any acquisition plans?
A: Continuously evaluating but cannot comment on specifics.

WORKPLACE

Q: Return-to-office position?
A: [CHRO] explained current policy: [SUMMARY]. Monitoring [FACTORS] with [X] weeks notice before changes.

Q: Addressing burnout?
A: [CHRO] outlined initiatives:
   - [INITIATIVE 1]
   - [INITIATIVE 2]

COMPENSATION

Q: Annual raise/bonus timing?
A: [CFO] confirmed:
   - Reviews: [DATE RANGE]
   - Decisions: [DATE]
   - Effective: [DATE]

UNANSWERED QUESTIONS
[NUMBER] questions answered in writing by [DATE] via [CHANNEL].

CLOSING
[CEO] encouraged ongoing dialogue and committed to [FREQUENCY] sessions.

NEXT TOWN HALL: [DATE]

Recording: [LINK]

Prepared by: [NAME]"""},
    ]
    
    # -------------------------------------------------------------------------
    # 8. TERMS OF SERVICE (8 variations)
    # -------------------------------------------------------------------------
    tos_templates = [
        {"instruction": "Write the introduction section of Terms of Service for a SaaS platform.",
         "output": """TERMS OF SERVICE

Last Updated: [DATE]

1. INTRODUCTION AND ACCEPTANCE

1.1 Agreement to Terms
These Terms of Service constitute a legally binding agreement between you ("User") and [COMPANY NAME] ("Company") governing your access to our website, applications, and services (collectively, "Services").

1.2 Acceptance
By accessing or using our Services, you acknowledge that you have read, understood, and agree to be bound by these Terms. If you do not agree, you may not use the Services.

1.3 Authority to Bind
If using on behalf of an organization, you represent authority to bind that entity to these Terms.

1.4 Eligibility
You must be at least 18 years old to use our Services.

1.5 Changes to Terms
We reserve the right to modify these Terms at any time. We will provide notice of material changes by posting updated Terms and updating the "Last Updated" date.

1.6 Additional Terms
Certain features may be subject to additional terms, incorporated by reference.

1.7 Privacy Policy
Our Privacy Policy at [URL] describes how we handle your information. By using Services, you also agree to our Privacy Policy."""},
        
        {"instruction": "Draft the user accounts section of Terms of Service.",
         "output": """USER ACCOUNTS AND RESPONSIBILITIES

2. ACCOUNT REGISTRATION

2.1 Account Creation
To access certain features, you must create an account. You agree to:
   a) Provide accurate, current, complete information
   b) Maintain and update your information
   c) Maintain security of your credentials
   d) Accept responsibility for all account activities
   e) Notify us immediately of unauthorized access

2.2 Account Security
You are solely responsible for maintaining credential confidentiality. You agree to:
   a) Use strong, unique passwords
   b) Enable two-factor authentication when available
   c) Not share credentials
   d) Log out from shared devices
   e) Report suspected security breaches

3. USER CONDUCT

3.1 Acceptable Use
Use Services only for lawful purposes. You agree not to:
   a) Violate any laws or third-party rights
   b) Upload malicious code or harmful content
   c) Attempt unauthorized access
   d) Interfere with Services
   e) Use for fraudulent purposes
   f) Harvest user information without consent
   g) Impersonate any person

3.2 Content Standards
Content must not:
   a) Be illegal, harmful, threatening, or abusive
   b) Infringe intellectual property rights
   c) Contain personal information without consent
   d) Be false or misleading
   e) Contain viruses or harmful code"""},
        
        {"instruction": "Write the intellectual property section of Terms of Service.",
         "output": """INTELLECTUAL PROPERTY AND CONTENT

4. COMPANY INTELLECTUAL PROPERTY

4.1 Ownership
The Services, including content, features, software, designs, and trademarks, are owned by [COMPANY NAME] or licensors and protected by intellectual property laws.

4.2 Limited License
Subject to compliance, we grant you a limited, non-exclusive, non-transferable, revocable license to access and use Services for your purposes.

4.3 Restrictions
You may not:
   a) Copy, modify, or distribute our Services or content
   b) Reverse engineer or decompile software
   c) Remove proprietary notices
   d) Use trademarks without permission
   e) Create derivative works
   f) Sublicense, sell, or transfer rights

5. USER CONTENT

5.1 Your Content
"User Content" means any data, text, files, or materials you upload or transmit.

5.2 Ownership
You retain all ownership rights in your User Content.

5.3 License Grant
By uploading, you grant us a worldwide, non-exclusive, royalty-free license to use, reproduce, modify, and display your Content solely to provide and improve Services. This license terminates when you delete your Content.

5.4 Representations
You warrant that:
   a) You own or have rights to your Content
   b) Your Content does not infringe third-party rights
   c) Your Content complies with these Terms

5.5 Feedback
Suggestions or feedback may be used without restriction or compensation."""},
        
        {"instruction": "Create the payment terms section of Terms of Service.",
         "output": """PAYMENT AND SUBSCRIPTION TERMS

6. FEES AND PAYMENT

6.1 Subscription Plans
Services offered through various plans as described on our pricing page.

6.2 Fees
You agree to pay all fees for your plan. All fees are:
   a) Stated in [USD]
   b) Exclusive of taxes unless stated
   c) Non-refundable except as provided
   d) Subject to change with [30] days notice

6.3 Payment Method
Provide a valid payment method. You authorize charges for all fees incurred.

6.4 Billing Cycle
   a) Monthly: billed on same date each month
   b) Annual: billed on subscription anniversary

6.5 Failed Payments
If payment fails, we will attempt again, notify you, and may suspend access after [X] days.

7. SUBSCRIPTION MANAGEMENT

7.1 Free Trials
At trial end:
   a) Subscription converts to paid unless cancelled
   b) Payment method will be charged

7.2 Changes
   a) Upgrades: immediate with prorated billing
   b) Downgrades: effective at next billing cycle

7.3 Cancellation
Cancel anytime through account settings. Upon cancellation:
   a) Access retained until billing period end
   b) No partial-period refunds
   c) Data retained [X] days, then deleted

8. TAXES
You are responsible for applicable taxes."""},
        
        {"instruction": "Write the disclaimer and limitation of liability section.",
         "output": """DISCLAIMERS AND LIMITATIONS

9. DISCLAIMER OF WARRANTIES

9.1 "As Is" Basis
SERVICES PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING MERCHANTABILITY, FITNESS FOR PARTICULAR PURPOSE, OR NON-INFRINGEMENT.

9.2 No Guarantee
We do not warrant:
   a) Services will be uninterrupted or error-free
   b) Results will be accurate or reliable
   c) Quality will meet expectations
   d) Defects will be corrected

9.3 Third-Party Content
We are not responsible for third-party content or services.

10. LIMITATION OF LIABILITY

10.1 Exclusion of Damages
TO MAXIMUM EXTENT PERMITTED, [COMPANY NAME], ITS DIRECTORS, EMPLOYEES, OR AFFILIATES SHALL NOT BE LIABLE FOR INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING LOSS OF PROFITS, DATA, USE, OR GOODWILL.

10.2 Cap on Liability
TOTAL LIABILITY SHALL NOT EXCEED THE GREATER OF: (A) AMOUNT PAID IN TWELVE MONTHS PRECEDING THE CLAIM; OR (B) ONE HUNDRED DOLLARS ($100).

10.3 Exceptions
Some jurisdictions prohibit certain exclusions. In such jurisdictions, liability limited to greatest extent permitted.

10.4 Basis of Bargain
YOU ACKNOWLEDGE THESE LIMITATIONS ARE ESSENTIAL ELEMENTS AND WE WOULD NOT PROVIDE SERVICES WITHOUT THEM."""},
        
        {"instruction": "Draft the termination section of Terms of Service.",
         "output": """TERMINATION AND SUSPENSION

11. TERMINATION

11.1 By You
Terminate your account anytime through account settings or by contacting [EMAIL].

11.2 By Us
We may terminate or suspend your account immediately, without notice, for any reason, including:
   a) Breach of these Terms
   b) Failure to pay fees
   c) Fraudulent or illegal activity
   d) Misuse of Services
   e) Security risk

11.3 Effect of Termination
Upon termination:
   a) Right to use Services ceases immediately
   b) You must cease all use
   c) Data may be deleted after [X] days
   d) Outstanding fees become due
   e) Surviving provisions remain effective

11.4 Data Export
Prior to termination, you may export data through Services' export functionality.

11.5 Reinstatement
At our sole discretion, subject to additional terms or fees.

12. SUSPENSION

12.1 Right to Suspend
We may suspend access for:
   a) Suspected Terms violation
   b) Security concerns
   c) Maintenance
   d) Legal requirements
   e) Non-payment

12.2 Notice
When possible, we provide notice before suspension. Immediate suspension may occur without notice in emergencies."""},
        
        {"instruction": "Write the dispute resolution section of Terms of Service.",
         "output": """DISPUTE RESOLUTION

13. GOVERNING LAW
These Terms governed by laws of the State of [STATE], without regard to conflict of law provisions.

14. ARBITRATION AGREEMENT

14.1 Agreement to Arbitrate
You and [COMPANY NAME] agree disputes shall be resolved by binding arbitration, rather than court, except either party may seek equitable relief for IP infringement.

14.2 Arbitration Rules
Conducted by [JAMS/AAA] under current rules in [CITY, STATE] or via telephone/video.

14.3 Process
   a) One arbitrator selected per applicable rules
   b) Decision final and binding
   c) Judgment may be entered in any court

14.4 Costs
Each party bears its own attorneys' fees. Arbitration fees shared equally unless arbitrator determines otherwise.

15. CLASS ACTION WAIVER

15.1 No Class Actions
YOU AND [COMPANY NAME] AGREE EACH MAY BRING CLAIMS ONLY IN INDIVIDUAL CAPACITY, NOT AS PLAINTIFF OR CLASS MEMBER IN ANY CLASS OR REPRESENTATIVE ACTION.

15.2 No Consolidation
Unless both parties agree, arbitrator may not consolidate claims or preside over representative proceedings.

16. TIME LIMITATION
Any claim must be filed within one (1) year after cause of action arose.

17. EQUITABLE RELIEF
Either party may seek injunctive relief in court to prevent IP infringement."""},
        
        {"instruction": "Create the general provisions section of Terms of Service.",
         "output": """GENERAL PROVISIONS

18. ENTIRE AGREEMENT
These Terms, together with Privacy Policy and incorporated agreements, constitute the entire agreement and supersede all prior agreements.

19. SEVERABILITY
If any provision found unenforceable, it shall be limited or eliminated to minimum extent necessary; remaining provisions remain effective.

20. WAIVER
Our failure to enforce any right shall not constitute waiver. Any waiver must be in writing signed by authorized representative.

21. ASSIGNMENT
You may not assign these Terms without our consent. We may assign without restriction.

22. NOTICES
   a) To You: Via email, posting on Services, or reasonable means
   b) To Us: [ADDRESS] or [EMAIL]
   c) Effective upon receipt or posting

23. FORCE MAJEURE
Neither party liable for failure due to circumstances beyond reasonable control, including acts of God, natural disasters, war, terrorism, strikes, or shortages.

24. INDEPENDENT CONTRACTORS
Parties are independent contractors. Nothing creates partnership, joint venture, agency, or employment.

25. THIRD-PARTY BENEFICIARIES
These Terms confer no rights on third parties unless expressly stated.

26. HEADINGS
Section headings for convenience only; shall not affect interpretation.

27. CONTACT INFORMATION
[COMPANY NAME]
[ADDRESS]
Email: [EMAIL]
Phone: [PHONE]

BY USING OUR SERVICES, YOU ACKNOWLEDGE THAT YOU HAVE READ, UNDERSTOOD, AND AGREE TO BE BOUND BY THESE TERMS OF SERVICE."""},
    ]
    
    # Combine all templates
    dataset.extend(employment_templates)
    dataset.extend(nda_templates)
    dataset.extend(privacy_templates)
    dataset.extend(recommendation_templates)
    dataset.extend(email_templates)
    dataset.extend(sales_templates)
    dataset.extend(meeting_templates)
    dataset.extend(tos_templates)
    
    return dataset


# ============================================================================
# DATA FORMATTING AND TOKENIZATION
# ============================================================================

def format_example(example: Dict[str, str]) -> str:
    """Format a single example into instruction-response format."""
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"


def create_tokenized_dataset(dataset: List[Dict[str, str]], tokenizer, max_length: int = 512) -> Dataset:
    """Create a tokenized HuggingFace Dataset for training."""
    formatted_examples = [format_example(ex) for ex in dataset]
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        
        # For causal LM, labels = input_ids with padding masked as -100
        labels = []
        for input_ids in tokenized["input_ids"]:
            label = input_ids.copy()
            label = [-100 if token == tokenizer.pad_token_id else token for token in label]
            labels.append(label)
        
        tokenized["labels"] = labels
        return tokenized
    
    hf_dataset = Dataset.from_dict({"text": formatted_examples})
    tokenized_dataset = hf_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


# ============================================================================
# MODEL SETUP AND TRAINING
# ============================================================================

def setup_model_and_tokenizer(model_name: str, device: str):
    """Load the base model and tokenizer with LoRA configuration."""
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # GPT-2 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"Total parameters: {total_params:,}")
    
    return model, tokenizer


def train_model(model, tokenizer, train_dataset, output_dir: str, epochs: int = 3,
                batch_size: int = 2, learning_rate: float = 2e-4):
    """Train the model using the Trainer API."""
    logger.info("Setting up training arguments...")
    
    # Training arguments optimized for RTX 3060 6GB
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Reduced for faster updates
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,  # Use ratio instead of fixed steps
        logging_steps=5,
        save_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=True,
        seed=42,
        load_best_model_at_end=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training complete!")


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
    ]
    
    logger.info("\n" + "=" * 60)
    logger.info("TESTING MODEL GENERATION")
    logger.info("=" * 60)
    
    for i, instruction in enumerate(test_instructions, 1):
        logger.info(f"\n--- Test {i} ---")
        logger.info(f"Instruction: {instruction}")
        
        response = generate_text(model, tokenizer, instruction, device=device)
        
        logger.info(f"\nGenerated Response:\n{response[:1000]}...")
        logger.info("-" * 40)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 for English legal/business text generation using LoRA/PEFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--model_name", type=str, default="gpt2-medium", help="Base model name (gpt2, gpt2-medium, gpt2-large)")
    parser.add_argument("--output_dir", type=str, default="./legal_llm_finetuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (minimum 5 recommended)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--test_only", action="store_true", help="Test mode (3 epochs, 50 examples)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Creating legal/business text dataset...")
    dataset = create_legal_business_dataset()
    logger.info(f"Dataset size: {len(dataset)} examples")
    
    if args.test_only:
        logger.info("TEST MODE: Using 50 examples and 3 epochs for meaningful test")
        dataset = dataset[:50]
        args.epochs = 3
    
    model, tokenizer = setup_model_and_tokenizer(args.model_name, device)
    
    logger.info("Tokenizing dataset...")
    train_dataset = create_tokenized_dataset(dataset, tokenizer, args.max_length)
    logger.info(f"Tokenized dataset size: {len(train_dataset)}")
    
    logger.info("\nVerifying trainable parameters:")
    model.print_trainable_parameters()
    
    train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    logger.info("\nTesting model generation...")
    test_generation(model, tokenizer, device)
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training complete! Model saved to: {args.output_dir}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()

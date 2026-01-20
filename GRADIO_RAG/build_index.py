#!/usr/bin/env python3
"""
Generate pre-built FAISS index with sample HR policies.
Run this ONCE on your local PC to create the faiss_index/ folder.
Then commit it to GitHub.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Sample HR policies content
SAMPLE_POLICIES = """
EMPLOYEE HANDBOOK - TECHCORP INC.
Last Updated: January 2025

================================================================================
SECTION 1: TIME OFF & LEAVE POLICIES
================================================================================

1.1 PAID TIME OFF (PTO)
- Full-time employees receive 15 days of PTO per year
- PTO accrues at a rate of 1.25 days per month
- Maximum PTO accumulation is 25 days
- Unused PTO can be carried over to the next year (up to 5 days)
- PTO requests must be submitted at least 2 weeks in advance
- Manager approval is required for all PTO requests

1.2 SICK LEAVE
- Employees receive 10 sick days per year
- Sick leave does not carry over to the next year
- A doctor's note is required for absences exceeding 3 consecutive days
- Sick leave can be used for personal illness or caring for immediate family

1.3 HOLIDAYS
The company observes the following paid holidays:
- New Year's Day (January 1)
- Martin Luther King Jr. Day
- Presidents Day
- Memorial Day
- Independence Day (July 4)
- Labor Day
- Thanksgiving Day and the day after
- Christmas Eve and Christmas Day

1.4 PARENTAL LEAVE
- Primary caregiver: 12 weeks paid leave
- Secondary caregiver: 4 weeks paid leave
- Leave must be taken within 12 months of birth or adoption

================================================================================
SECTION 2: WORKING HOURS & REMOTE WORK
================================================================================

2.1 STANDARD WORKING HOURS
- Core business hours: 9:00 AM to 5:00 PM, Monday through Friday
- Full-time employees work 40 hours per week
- Lunch break: 1 hour (unpaid)
- Two 15-minute paid breaks per day

2.2 FLEXIBLE WORK ARRANGEMENTS
- Flex-time available with manager approval
- Core hours (must be present): 10:00 AM to 3:00 PM
- Start time can range from 7:00 AM to 10:00 AM

2.3 REMOTE WORK POLICY
- Hybrid work: Up to 3 days remote work per week
- Full remote work requires VP approval
- Employees must be available during core business hours
- Home office stipend: $500 one-time setup allowance
- Monthly internet allowance: $50

2.4 OVERTIME
- Non-exempt employees receive 1.5x pay for hours over 40/week
- Overtime must be pre-approved by manager
- Exempt employees are not eligible for overtime pay

================================================================================
SECTION 3: COMPENSATION & BENEFITS
================================================================================

3.1 PAY SCHEDULE
- Employees are paid bi-weekly (every two weeks)
- Direct deposit is mandatory
- Pay stubs available through the HR portal

3.2 HEALTH INSURANCE
Medical Coverage Options:
- PPO Plan: $150/month employee contribution
- HMO Plan: $100/month employee contribution
- HSA-compatible High Deductible Plan: $75/month + company HSA contribution

Coverage includes:
- Medical, dental, and vision
- Prescription drug coverage
- Mental health services
- Company pays 80% of premium costs

3.3 401(K) RETIREMENT PLAN
- Company matches 100% of contributions up to 4% of salary
- Additional 50% match on next 2% of salary
- Immediate vesting on employee contributions
- Company match vests over 3 years (33% per year)
- Eligible to participate after 90 days of employment

3.4 LIFE INSURANCE
- Basic life insurance: 2x annual salary (company paid)
- Optional supplemental coverage available
- AD&D insurance included

3.5 OTHER BENEFITS
- Employee Assistance Program (EAP)
- Gym membership subsidy: $50/month
- Professional development budget: $2,000/year
- Tuition reimbursement: Up to $5,250/year
- Commuter benefits (pre-tax)

================================================================================
SECTION 4: WORKPLACE POLICIES
================================================================================

4.1 DRESS CODE
- Business casual attire is standard
- Casual Fridays allow jeans and company t-shirts
- Client meetings require business professional attire
- Safety equipment required in designated areas

4.2 ATTENDANCE
- Employees must notify supervisor of absence by 9:00 AM
- Excessive absenteeism may result in disciplinary action
- Three or more consecutive unexcused absences may result in termination

4.3 CODE OF CONDUCT
- Maintain professional behavior at all times
- Respect colleagues and maintain a harassment-free workplace
- Protect confidential company information
- Report ethics violations to HR or anonymous hotline

4.4 TECHNOLOGY USE
- Company equipment for business use primarily
- Limited personal use permitted
- No installation of unauthorized software
- All activity on company systems may be monitored

================================================================================
SECTION 5: PERFORMANCE & DEVELOPMENT
================================================================================

5.1 PERFORMANCE REVIEWS
- Annual performance reviews in December
- Mid-year check-ins in June
- Reviews tied to merit increases and bonuses

5.2 PROFESSIONAL DEVELOPMENT
- Annual training budget: $2,000 per employee
- Conference attendance with manager approval
- Internal mentorship program available
- Lunch-and-learn sessions monthly

================================================================================
SECTION 6: PROCEDURES
================================================================================

6.1 HOW TO REQUEST TIME OFF
1. Log into the HR portal
2. Navigate to "Time Off" section
3. Select dates and type of leave
4. Submit request to manager
5. Await approval notification

6.2 HOW TO SUBMIT EXPENSES
1. Collect all receipts
2. Log into expense system within 30 days
3. Categorize and describe each expense
4. Attach receipt images
5. Submit for manager approval
6. Reimbursement within 2 pay cycles

6.3 HOW TO REPORT ISSUES
- HR concerns: Contact HR department or use anonymous hotline
- IT issues: Submit ticket through IT portal
- Facilities issues: Email facilities@company.com
- Safety concerns: Report immediately to supervisor and Safety team
"""

def main():
    print("🚀 Building FAISS index with sample HR policies...")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ".", " "]
    )
    
    # Create documents
    docs = [Document(page_content=SAMPLE_POLICIES, metadata={"source": "HR_Handbook"})]
    chunks = text_splitter.split_documents(docs)
    print(f"📄 Created {len(chunks)} chunks")
    
    # Initialize embeddings
    print("🤖 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Build FAISS index
    print("🔨 Building FAISS vector store...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # Save to disk
    print("💾 Saving index to faiss_index/...")
    vector_db.save_local("faiss_index")
    
    print("✅ Done! faiss_index/ folder created.")
    print("\n📋 Next steps:")
    print("1. git add faiss_index/")
    print("2. git commit -m 'Add pre-built FAISS index'")
    print("3. git push")

if __name__ == "__main__":
    main()

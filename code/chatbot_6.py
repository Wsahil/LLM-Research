from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import numpy as np

# Step 1: Sample data
data = [
    {"course_name": "IDS 655 - Global Information Technology Abroad", "description": "0 credit(s) Upon sufficient interest. Double Numbered with IDS 355. This course is the travel portion of Global Information Technology Abroad. Students enrolled in IDS 350/IDS 650 will register for this course as a requirement to travel when the trip is scheduled. Repeatable 2 time(s), 0 credits maximum."},

    {"course_name": "IST 603 - Global Financial Systems Architecture","description": "3 credit(s) At least 1x fall or spring. Double Numbered with IST 303. Structures of real-world information systems in money supply chain. Emphasis on large-scale banking organizations and their challenges in moving and processing millions of complex transactions worldwide. Additional work for graduate students."},

    {"course_name": "IDS 600 - Selected Topics", "description": "1-6 credit(s) Irregularly. Exploration of a topic (to be determined) not covered by the standard curriculum but of interest to faculty and students in a particular semester. Repeatable."},

    {"course_name": "IDS 601 - What’s the Big Idea: Technology Innovation", "description": "3 credit(s) At least 1x fall or spring. Double Numbered with IDS 301. Innovation in information technology. Learn how to develop, grow and vet ideas and build teams ready to create a product, service, or business. Additional work required of graduate students."},

    {"course_name": "IDS 602 - idea2Startup","description": "3 credit(s) At least 1x fall or spring. Double Numbered with IDS 302. Students establish an actionable plan for the launch of their own business. Focus on establishing a well-conceived, achievable, and actionable path to market. Additional work required of graduate students."},

    {"course_name": "IDS 650 - Global Information Technology Abroad", 
     "description": "3-6 credit(s) At least 1x fall or spring. Double Numbered with IDS 350. Explore the roles that information and digital technologies play in different national and geographic contexts. Learn how organizations in various countries use information technology for strategic advantage in an increasingly competitive global marketplace. Additional work for graduates. Repeatable 2 time(s), 12 credits maximum."},

    {"course_name": "IDS 660 - Information Technology Experience", 
     "description": "1-3 credit(s) At least 1x fall or spring. Double Numbered with IDS 360. Explore the roles that information and digital technologies play in different organizations. Learn how organizations use information technology for strategic advantage in an increasingly competitive global marketplace. Additional work for graduates. Repeatable 2 time(s), 6 credits maximum."},

    {"course_name": "IDS 690 - Independent Study", 
     "description": "1-6 credit(s). Exploration of a problem, or problems, in depth. Individual independent study upon a plan submitted by the student. Admission by consent of supervising instructor(s) and the department. Repeatable."},

    {"course_name": "IST 500 - Selected Topics", 
     "description": "1-6 credit(s) Upon sufficient interest. Exploration of a topic (to be determined) not covered by the standard curriculum but of interest to faculty and students in a particular semester. Repeatable."},

    {"course_name": "IST 511 - Cultural Foundations of Information Studies", 
     "description": "3 credit(s) At least 1x fall or spring. Survey of the professional, social, ethical, and legal issues affecting information service professionals and organizations and prepares students to deal with these issues and work with a diverse community."},

    {"course_name": "IST 523 - Graphic Design for the Web", 
     "description": "3 credit(s) At least 1x fall or spring. Learn basic and advanced website design principles utilizing Adobe Photoshop and Flash, with emphasis on typography, color theory, and layout. Understand and practice Flash Actionscript basics to create animation and dynamic web applications."},

    {"course_name": "IST 564 - Accessible Library & Information Services", 
     "description": "3 credit(s) At least 1x fall or spring. Includes strategies for planning library programs and services that are inclusive of the information needs of PK-12th grade students with disabilities."},

    {"course_name": "IST 600 - Selected Topics", 
     "description": "1-6 credit(s) Upon sufficient interest. Exploration of a topic (to be determined) not covered by the standard curriculum but of interest to faculty and students in a particular semester. Repeatable."},

    {"course_name": "IST 602 - Digital Forensics", 
     "description": "3 credit(s) At least 1x fall or spring. Examination of information technology to establish probative information. Fundamentals of the forensic process, evidence handling, and quality assurance as these apply to digital forensics."},

    {"course_name": "IST 604 - Cataloging of Information Resources", 
     "description": "3 credit(s) Upon sufficient interest. Cataloging rules, standards, and metadata schemes; bibliographic utilities; formats of print and non-print materials; cataloging software; management issues. PREREQ: IST 616."},

    {"course_name": "IST 605 - Reference and Information Literacy Services", 
     "description": "3 credit(s) At least 1x fall or spring. The discovery and use of print and electronic resources and delivery of services in libraries to meet information needs of varied patron communities in a broad range of contexts."},

    {"course_name": "IST 607 - Digital Humanities for Librarians, Archivists, & Cultural Heritage Workers", 
     "description": "3 credit(s) At least 1x fall or spring. Introduction to Digital Humanities (DH) in Libraries, Archives, Museums, and other cultural institutions, with a focus on how information professionals can make use of DH methods and approaches and support and sustain DH work in their communities."},

    {"course_name": "IST 608 - Blockchain Management", 
     "description": "3 credit(s) At least 1x fall or spring. Double Numbered with IST 408. Students complete distributed ledger labs before developing, implementing, and ‘demo or die’ sharktanking their own new blockchain project. Blockchain concepts such as decentralization, smart contracts, trust, and consensus governance are discussed. Additional work for graduate students."},

    {"course_name": "IST 611 - Information Technologies in Educational Organizations", 
     "description": "3 credit(s) At least 1x fall or spring. Information and communication technologies for teaching and learning in all types of libraries: practice with tools along with consideration of pedagogy (instructional design and evaluation of tools for learning objectives) and context (ethics, policies, legislation)."},

    {"course_name": "IST 612 - Youth Services in Libraries and Information Centers", 
     "description": "3 credit(s) At least 1x fall or spring. Theories, practices, media, literature and emerging trends of youth services from preschool to high school are explored. A broad range of competencies necessary to work with youth in a variety of library settings are presented."},

    {"course_name": "IST 613 - Library Planning, Marketing, and Assessment", 
     "description": "3 credit(s) At least 1x fall or spring. User-focused planning, marketing, and assessment of activities that support core functions of libraries, such as collection development, systems, and public services. PREREQ: IST 605."},

    {"course_name": "IST 614 - Information Technology Management and Policy", 
     "description": "3 credit(s) Every semester. Basic ideas, concepts and perspectives of management and policy as they apply to the information professions. Students learn to understand and apply these principles to improve organizational effectiveness and promote digital transformation."},

    {"course_name": "IST 615 - Cloud Management", 
     "description": "3 credit(s) At least 1x fall or spring. Cloud services creation and management. Practical experience in using, creating, and managing digital services across data centers and hybrid clouds. Strategic choices for cloud digital service solutions across open data centers and software defined networks."},

    {"course_name": "IST 616 - Information Resources: Organization and Access", 
     "description": "3 credit(s) At least 1x fall or spring. Introduction to theories, tools, and standards for information organization and access, including cataloging rules and formats, content analysis, indexing, classification, and fundamentals of information retrieval systems."},

    {"course_name": "IST 617 - Motivational Aspects of Information Use", 
     "description": "3 credit(s) Upon sufficient interest. Theories of motivation and behavior affecting information use in learning, workplace, and virtual environments. Emphasis on applying motivational theories and models to management practices in information organizations and to the design of information resources and presentations."},

    {"course_name": "IST 618 - Information Policy", 
     "description": "3 credit(s) Every semester. Public policy issues that affect the information, library, and telecommunication sectors. These include privacy and security, intellectual property, freedom of expression, communications access, and public sector information. Application of economic, legal, and political science concepts to policy analysis."},

    {"course_name": "IST 619 - Economics of Digital Transformation", 
     "description": "3 credit(s) At least 1x fall or spring. Double Numbered with IST 419. Application of economic principles to the information industries, online marketplaces, and information management. Discussion of how advances in digital technology transform society. Topics include revenue, cost, industrial organization, market power, public goods, and optimal decision-making. Additional work for graduate students."},

    {"course_name": "IST 620 - Advanced Topics in Information Innovation", 
     "description": "3 credit(s) At least 1x fall or spring. Critical and emerging issues in information innovation. The course presents durable concepts of innovation around a specific innovation in the information field. The innovation is used as context for readings and project."},

    {"course_name": "IST 621 - Information Management and Technology", 
     "description": "3 credit(s) Every semester. Information and technology management overview with a focus on digital transformation. How information and technology managers create organizational, technological, and personal capabilities to succeed in a rapidly changing digital world."},

    {"course_name": "IST 622 - Introduction to Preservation of Cultural Heritage", 
     "description": "3 credit(s) At least 1x fall or spring. Introduction to field of preservation of cultural heritage, including institutions, contexts and methodologies, concepts of place and culture, objects and resources for study; emphasis also on role of digital applications. Requires research project and presentation."},

    {"course_name": "IST 623 - Introduction to Information Security", 
     "description": "3 credit(s) Every semester. Basic concepts and technologies of information security. Students who successfully complete this course will have a comprehensive overview of information security with some hands-on experience."},

    {"course_name": "IST 624 - Preservation of Library and Archival Collections", 
     "description": "3 credit(s) At least 1x fall or spring. Introduction to preserving library and archival collections, including paper, audio-visual, and digital objects."},

    {"course_name": "IST 625 - Enterprise Risk Management", 
     "description": "3 credit(s) At least 1x fall or spring. A multidisciplinary perspective of risk assessment, modeling, and management. Topics include: concepts of personal accountability versus governance and policy; how organizations define and measure risk and loss; and plan for contingencies."},

    {"course_name": "IST 626 - Information Justice & Community Engagement", 
     "description": "3 credit(s) At least 1x fall or spring. Double Numbered with IST 426. Course examines interrelation of social justice and community engagement in online and naturalistic communities. Areas of emphasis will explore how gender, race, class, and community location (On and Off-line) affect and are affected by information. Additional work for graduate students."}



















]

# Step 2: Initialize Pinecone
api_key = "pcsk_2xrWik_8aW44s4muE68taxW7YG2NztkQiTTDjPifHYcdyxDtCDbWaRR1Vq2Q2HrBSVEok5"
region = "us-east-1"

pc = Pinecone(api_key=api_key)

# Step 3: Checks if the index exists and create if not
index_name = "ischool-courses"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension for the chosen SentenceTransformer model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=region
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Step 4: Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 5: Generate embeddings for course descriptions and upsert to Pinecone
for idx, item in enumerate(data):
    embedding = model.encode(item["course_name"]).tolist()
    metadata = {
        "course_name": item["course_name"],
        "description": item["description"]
    }
    index.upsert([(f"course-{idx}", embedding, metadata)])

print("Data successfully added to Pinecone!")

# Step 6: Query Pinecone
def query_pinecone(query_text):
    query_embedding = model.encode(query_text).tolist()
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    return results

# Test the query
query = "Cloud Management"
result = query_pinecone(query)

print("Query Results:")
for match in result['matches']:
    print(f"Course Name: {match['metadata']['course_name']}, Score: {match['score']}")

# Student Opportunity Hub

Student Opportunity Hub is a modern web platform designed to help students discover internships, courses, hackathons, and project opportunities in one place.

Instead of searching through multiple websites and tabs, users can browse opportunities, search by title or organization, filter by category and mode, sort by deadlines, and manage opportunity listings through a simple and polished interface.

## Features

- Search opportunities by title or organization
- Browse internships, courses, hackathons, and projects
- Filter opportunities by category
- Filter by learning/work mode
- Sort opportunities by deadline
- Add new opportunities
- Edit existing opportunities
- Delete opportunities
- Input validation
- User-friendly error handling
- Opportunity statistics
- Persistent database storage
- Responsive design for different screen sizes
- Modern and student-friendly interface

## Tech Stack

### Backend

- Python
- FastAPI
- Uvicorn
- Pydantic

### Database

- SQLite
- SQLAlchemy

### Frontend

- HTML
- CSS
- JavaScript

## How It Works

The application uses FastAPI as the backend and SQLite for persistent data storage.

Users interact with the frontend, while the backend handles requests and communicates with the database.

The application supports complete data management:

**Create → Read → Update → Delete**

All opportunity information is stored in the database rather than temporary browser storage.

## Project Structure

```text
student-opportunity-hub/
│
├── main.py
├── requirements.txt
└── opportunities.db

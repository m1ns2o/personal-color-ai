<template>
  <div class="entry-page">
    <header class="header">
      <h1>AI 퍼스널 진단</h1>
      <p>원하는 진단을 선택해주세요</p>
    </header>

    <main class="main-content">
      <div class="service-cards">
        <router-link to="/personal-color" class="service-card">
          <div class="card-icon">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="64"
              height="64"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <circle cx="8.5" cy="8.5" r="1.5"></circle>
              <polyline points="21 15 16 10 5 21"></polyline>
            </svg>
          </div>
          <h2>퍼스널 컬러 진단</h2>
          <p>나에게 어울리는 계절 컬러를 찾아보세요</p>
        </router-link>

        <router-link to="/face-shape" class="service-card">
          <div class="card-icon">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="64"
              height="64"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M12 2a10 10 0 0 0-10 10 10 10 0 0 0 10 10 10 10 0 0 0 10-10A10 10 0 0 0 12 2z"></path>
              <path d="M8 14s1.5 2 4 2 4-2 4-2"></path>
              <line x1="9" y1="9" x2="9.01" y2="9"></line>
              <line x1="15" y1="9" x2="15.01" y2="9"></line>
            </svg>
          </div>
          <h2>얼굴형 진단</h2>
          <p>나의 얼굴형을 분석하고 스타일을 추천받으세요</p>
        </router-link>
      </div>
    </main>
  </div>
</template>

<style scoped>
.entry-page {
  min-height: 100vh;
  background: #ffffff;
  padding: 4rem 2rem;
}

.header {
  text-align: center;
  margin-bottom: 4rem;
}

.header h1 {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
  font-weight: 700;
  color: #000000;
}

.header p {
  font-size: 1.2rem;
  color: #666666;
}

.main-content {
  max-width: 1000px;
  margin: 0 auto;
}

.service-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 2rem;
  padding: 1rem;
}

.service-card {
  background: white;
  border-radius: 12px;
  padding: 3rem 2rem;
  text-decoration: none;
  color: inherit;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  border: 2px solid #e0e0e0;
}

.service-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 12px 24px rgba(74, 144, 226, 0.2);
  border-color: #4a90e2;
}

.card-icon {
  width: 100px;
  height: 100px;
  background: #4a90e2;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 1.5rem;
  transition: all 0.3s ease;
}

.service-card:hover .card-icon {
  background: #357abd;
  transform: scale(1.05);
}

.card-icon svg {
  stroke: white;
}

.service-card h2 {
  font-size: 1.8rem;
  margin-bottom: 0.5rem;
  color: #000000;
  font-weight: 600;
}

.service-card p {
  color: #666;
  line-height: 1.6;
  font-size: 1rem;
}

@media (max-width: 768px) {
  .entry-page {
    padding: 2rem 1rem;
  }

  .header h1 {
    font-size: 2rem;
  }

  .header p {
    font-size: 1rem;
  }

  .service-cards {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }

  .service-card {
    padding: 2rem 1.5rem;
  }

  .service-card:hover {
    transform: none;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    border-color: #e0e0e0;
  }

  .service-card:hover .card-icon {
    background: #4a90e2;
    transform: none;
  }

  .card-icon {
    width: 80px;
    height: 80px;
  }

  .card-icon svg {
    width: 48px;
    height: 48px;
  }

  .service-card h2 {
    font-size: 1.5rem;
  }
}
</style>

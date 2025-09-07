CREATE TABLE test.crime_data (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  occ_date DATE NOT NULL,
  occ_year INT NOT NULL,
  occ_month VARCHAR(255) NOT NULL,
  occ_day INT NOT NULL,
  occ_dow VARCHAR(20),
  occ_hour INT,
  mci_category VARCHAR(100),
  offence VARCHAR(255),
  neighborhood VARCHAR(150),
  premises_type VARCHAR(150),
  embedding VECTOR(768),  -- Gemini embedding

  INDEX idx_year_month (occ_year, occ_month),
  INDEX idx_category (mci_category),
  INDEX idx_neighborhood (neighborhood),

  VECTOR INDEX vec_embedding ((VEC_COSINE_DISTANCE(embedding)))
);
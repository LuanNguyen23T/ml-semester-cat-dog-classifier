from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Pet Breed Classifier"
    API_V1_STR: str = "/api/v1"
    
    DATABASE_URL: str = ""
    POSTGRES_USER: str = "admin"
    POSTGRES_PASSWORD: str = "secret"
    POSTGRES_DB: str = "pet_classifier"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: str = "5432"

    CLOUDINARY_CLOUD_NAME: str = ""
    CLOUDINARY_API_KEY: str = ""
    CLOUDINARY_API_SECRET: str = ""
    CLOUDINARY_URL: str = ""
    AI_DEVICE: str = "auto"

    class Config:
        env_file = ".env"

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


settings = Settings()

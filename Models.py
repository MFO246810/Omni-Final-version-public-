import enum
import uuid
from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy import Text, Boolean, Enum, TIMESTAMP, func, Column, DateTime
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

class Base(DeclarativeBase):
    pass

class DataType(enum.Enum):
    pdf = "pdf"
    epub = "epub"
    url = "url"
    txt = "txt"

class Active_Models(Base):
    __tablename__ = "Active_Models"

    ID: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    User_ID: Mapped[str] = mapped_column(ForeignKey("User.User_ID"))
    Model_Name: Mapped[str] = mapped_column(String(30))
    Model_AgenticPrompt: Mapped[str] = mapped_column(Text)
    Is_Active: Mapped[bool] = mapped_column(Boolean, default=False)
    User = relationship("User", back_populates="User_Models")

    def __repr__(self) -> str:
        return f"Model(ID={self.ID!r}, Model Name={self.Model_Name!r}, Agent Prompt={self.Model_AgenticPrompt!r})"

class Collections(Base):
    __tablename__ = "Collections"

    Collections_Name: Mapped[str] = mapped_column(String(64), primary_key=True)
    Collections_Title: Mapped[str] = mapped_column(String(64))
    Data_Type: Mapped[DataType] = mapped_column(Enum(DataType), nullable=False)
    Source_Original: Mapped[str] = mapped_column(Text, nullable=False)
    Created_AT:Mapped[DateTime] = mapped_column(
        TIMESTAMP, server_default=func.current_timestamp(), nullable=False
    )

    def __repr__(self) -> str:
        return f"Collection Name: {self.Collection_name!r}"

class User(Base):
    __tablename__ = "User"

    User_ID: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    User_Name: Mapped[str] = mapped_column(String(30))
    Password: Mapped[str] = mapped_column(String(30))
    User_Models = relationship("Active_Models", back_populates="User")

class Model_Collections(Base):
     __tablename__ = "Model_Collections"

     Model_ID: Mapped[str] = mapped_column(ForeignKey("Active_Models.ID"), primary_key=True)
     Collection_Name: Mapped[str] = mapped_column(ForeignKey("Collections.Collections_Name"), primary_key=True)
     
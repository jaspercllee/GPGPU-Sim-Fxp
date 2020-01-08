#pragma once
#include <iostream>
#include <sstream>
#include <string>
#include <cstdint>

namespace spirv {

template<class DestTy, class FromTy>
inline DestTy* dyn_cast(FromTy* F){
  return dynamic_cast<DestTy*>(F);
}

// Type related
enum class TypeTag {
  Integer,
  Integer8,
  Integer16,
  Float,
  FixedPoint
};

struct TypeBase {
  TypeTag Tag;

  unsigned Size;
  inline unsigned MaxVal(){
    return (unsigned)((1 << Size) - 1);
  }

  TypeBase(TypeTag TG, unsigned size) : 
    Tag(TG),
    Size(size) {}

  virtual ~TypeBase(){}
};

struct FxpType : public TypeBase {
  unsigned Width;
  int Exponent;
  // FIXME: Should we save this?
  unsigned Metadata;
  bool Signed;

  // FIXME: Share these mask with other fixed point class
  constexpr static
  unsigned FXP_MASK_ACCU_SIGNED = 0x01;
  
  constexpr static
  unsigned FXP_MASK_EXP_SIGNED = 0x02;

  FxpType() : 
    TypeBase(TypeTag::FixedPoint, 0){}
  FxpType(unsigned width, int exp, unsigned meta) :
    TypeBase(TypeTag::FixedPoint, width),
    Width(width), Exponent(exp), Metadata(meta),
    Signed(meta & FXP_MASK_ACCU_SIGNED){}

  static
  FxpType* Create(unsigned width, unsigned raw_exp, unsigned meta){
    int exp = (int)raw_exp;
    if(meta & FXP_MASK_EXP_SIGNED)
      exp = -exp;
    return new FxpType(width, exp, meta);
  }

  void print(std::ostream& OS){
    OS << "fxp"
       << "_" << Width
       << "_" << Exponent
       << "_" << Metadata;
  }
};

struct IntType : public TypeBase {
  IntType() : 
    TypeBase(TypeTag::Integer, 32){}
  void print(std::ostream& OS){ OS << "int"; }
};
struct Int8Type : public TypeBase {
  Int8Type() : 
    TypeBase(TypeTag::Integer8, 8){}
  void print(std::ostream& OS){ OS << "int8"; }
};
struct Int16Type : public TypeBase {
  Int16Type() : 
    TypeBase(TypeTag::Integer16, 16){}
  void print(std::ostream& OS){ OS << "int16"; }
};
struct FloatType : public TypeBase {
  FloatType() : 
    TypeBase(TypeTag::Float, 32){}
  void print(std::ostream& OS){ OS << "float"; }
};

template<unsigned W>
struct FxpWidth {};

template<>
struct FxpWidth<8> {
  using type = uint8_t;
  constexpr static uint8_t max = UINT8_MAX;
};
template<>
struct FxpWidth<16> {
  using type = uint16_t;
  constexpr static uint16_t max = UINT16_MAX;
};
template<>
struct FxpWidth<32> {
  using type = uint32_t;
  constexpr static uint32_t max = UINT32_MAX;
};

template<unsigned Width>
struct IntWidth {};

template<>
struct IntWidth<16> {
  using type = Int16Type;
  using value_type = int16_t;
};
template<>
struct IntWidth<32> {
  using type = IntType;
  using value_type = int32_t;
};
template<>
struct IntWidth<8> {
  using type = Int8Type;
  using value_type = int8_t;
};

template<unsigned Width>
struct UIntWidth {};

template<>
struct UIntWidth<16> {
  using type = Int16Type;
  using value_type = uint16_t;
};
template<>
struct UIntWidth<32> {
  using type = IntType;
  using value_type = uint32_t;
};
template<>
struct UIntWidth<8> {
  using type = Int8Type;
  using value_type = uint8_t;
};
} // end namespace spirv

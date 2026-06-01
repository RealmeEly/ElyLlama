// ReSharper disable CppDFAUnreachableFunctionCall
// ReSharper disable CppNonExplicitConvertingConstructor
// ReSharper disable CppNonExplicitConversionOperator
// ReSharper disable CppParameterMayBeConst
#ifndef FP16_HPP
#define FP16_HPP

#include <bit>
#include <cstdint>
#include <iostream>

// ReSharper disable once CppInconsistentNaming
class float16_t { // NOLINT(*-identifier-naming)
private:
  uint16_t bits;

  struct FromBitsT {};

  constexpr float16_t(uint16_t b, FromBitsT) : bits(b) {}

  // Convert float32 to float16 (with rounding)
  static constexpr uint16_t float2Fp16Bits(float f) {
    const auto fp32_uint32 = std::bit_cast<uint32_t>(f);
    const uint32_t sign = (fp32_uint32 >> 31) & 0x1;
    const uint32_t exp = (fp32_uint32 >> 23) & 0xFF;
    const uint32_t mant = fp32_uint32 & 0x7FFFFF;

    int32_t exp_adj = static_cast<int32_t>(exp) - 127 + 15;
    if (exp_adj >= 31) { // +∞ / -∞
      return (sign << 15) | 0x7C00;
    }
    if (exp_adj <= 0) { // Subnormal or zero
      const uint32_t mant_with_hidden = (exp != 0) ? (mant | 0x800000) : mant;
      const int32_t shift = 1 - exp_adj;
      if (shift > 24) {
        return static_cast<uint16_t>(sign << 15); // underflow to zero
      }
      const uint32_t mant_shifted = mant_with_hidden >> (shift + 13);
      const uint32_t round_bit = (mant_with_hidden >> (shift + 12)) & 1;
      const uint32_t sticky = mant_with_hidden & ((1U << (shift + 12)) - 1);
      const uint32_t round = ((round_bit != 0U) && ((sticky != 0U) || ((mant_shifted & 1) != 0U))) ? 1 : 0;
      return (sign << 15) | static_cast<uint16_t>(mant_shifted + round);
    }
    // Normal
    const uint32_t mant_trunc = mant >> 13;
    const uint32_t round_bit = (mant >> 12) & 1;
    const uint32_t sticky = mant & 0xFFF;
    const uint32_t round = ((round_bit != 0U) && ((sticky != 0U) || ((mant_trunc & 1) != 0U))) ? 1 : 0;
    uint32_t combined_mant = mant_trunc + round;
    if (combined_mant >= 0x400) {
      combined_mant = 0;
      exp_adj++;
      if (exp_adj >= 31) {
        return (sign << 15) | 0x7C00;
      }
    }
    return (sign << 15) | (exp_adj << 10) | static_cast<uint16_t>(combined_mant);
  }

  // Convert float16 to float32
  static constexpr float fp16Bits2Float(uint16_t fp16_bits) {
    const uint32_t sign = (fp16_bits >> 15) & 0x1;
    uint32_t exp = (fp16_bits >> 10) & 0x1F;
    uint32_t mant = fp16_bits & 0x3FF;

    uint32_t fp32_uint32 = 0;
    if (exp == 0) {
      if (mant == 0) {
        fp32_uint32 = sign << 31; // zero
      } else {
        // subnormal: normalize
        const int shift = __builtin_clz(mant) - 21; // clz(10-bit mant) → normalize
        mant <<= shift;
        exp = 1 - shift;
        fp32_uint32 = (sign << 31) | ((exp + 127 - 15) << 23) | ((mant & 0x3FF) << 13);
      }
    } else if (exp == 31) {
      fp32_uint32 = (sign << 31) | 0x7F800000 | ((mant != 0U) ? 0x400000 : 0);
    } else {
      const uint32_t exp32 = exp + (127 - 15);
      fp32_uint32 = (sign << 31) | (exp32 << 23) | (mant << 13);
    }
    return std::bit_cast<float>(fp32_uint32);
  }

public:
  // NOLINTBEGIN(*-explicit-constructor)
  constexpr float16_t() : bits(0) {}

  // ReSharper disable once CppDFAUnreachableFunctionCall
  constexpr float16_t(float f) : bits(float2Fp16Bits(f)) {}

  constexpr float16_t(double d) : float16_t(static_cast<float>(d)) {}

  constexpr float16_t(int8_t d) : float16_t(static_cast<float>(d)) {}

  constexpr float16_t(int16_t d) : float16_t(static_cast<float>(d)) {}

  constexpr float16_t(int32_t d) : float16_t(static_cast<float>(d)) {}

  constexpr float16_t(int64_t d) : float16_t(static_cast<float>(d)) {}

  static constexpr float16_t fromBits(uint16_t b) { return float16_t{b, FromBitsT{}}; }

  // Conversion
  constexpr operator float() const { return fp16Bits2Float(bits); }
  explicit constexpr operator double() const { return static_cast<float>(*this); }

  [[nodiscard]] constexpr uint16_t getBits() const { return bits; }

  // Negation
  constexpr float16_t operator-() const { return fromBits(bits ^ 0x8000); }

  // Arithmetic (done via float)
  float16_t& operator+=(const float16_t& rhs) {
    *this = static_cast<float>(*this) + static_cast<float>(rhs);
    return *this;
  }

  float16_t& operator-=(const float16_t& rhs) {
    *this = static_cast<float>(*this) - static_cast<float>(rhs);
    return *this;
  }

  float16_t& operator*=(const float16_t& rhs) {
    *this = static_cast<float>(*this) * static_cast<float>(rhs);
    return *this;
  }

  float16_t& operator/=(const float16_t& rhs) {
    *this = static_cast<float>(*this) / static_cast<float>(rhs);
    return *this;
  }

  // Constants
  static constexpr float16_t zero() { return fromBits(0x0000); }
  static constexpr float16_t negZero() { return fromBits(0x8000); }
  static constexpr float16_t infinity() { return fromBits(0x7C00); }
  static constexpr float16_t negInfinity() { return fromBits(0xFC00); }
  static constexpr float16_t nan() { return fromBits(0x7E00); }

  // Classification
  [[nodiscard]] constexpr bool isNan() const { return (bits & 0x7FFF) > 0x7C00; }
  [[nodiscard]] constexpr bool isInf() const { return (bits & 0x7FFF) == 0x7C00; }
  [[nodiscard]] constexpr bool isFinite() const { return !isNan() && !isInf(); }
  [[nodiscard]] constexpr int sign() const { return ((bits & 0x8000) != 0) ? -1 : 1; }

  // NOLINTEND(*-explicit-constructor)
};

// Operators
inline float16_t operator+(float16_t a, float16_t b) {
  a += b;
  return a;
}

inline float16_t operator-(float16_t a, float16_t b) {
  a -= b;
  return a;
}

inline float16_t operator*(float16_t a, float16_t b) {
  a *= b;
  return a;
}

inline float16_t operator/(float16_t a, float16_t b) {
  a /= b;
  return a;
}

// Stream output
inline std::ostream& operator<<(std::ostream& os, const float16_t& f16) {
  if (f16.isNan()) {
    return os << "float16(NaN)";
  }
  if (f16.isInf()) {
    return os << "float16(" << (f16.sign() < 0 ? "-" : "") << "inf)";
  }
  return os << "float16(" << static_cast<float>(f16) << ")";
}

#if __cplusplus >= 202302L // c++23
template <>
struct std::numeric_limits<float16_t> {
public:
  static constexpr bool IS_SPECIALIZED = true;
  static constexpr bool IS_SIGNED = true;
  static constexpr bool IS_INTEGER = false;
  static constexpr bool IS_EXACT = false;
  static constexpr bool HAS_INFINITY = true;
  static constexpr bool HAS_QUIET_NA_N = true;
  static constexpr bool HAS_SIGNALING_NA_N = true;
  static constexpr float_denorm_style HAS_DENORM = denorm_present;
  static constexpr bool HAS_DENORM_LOSS = false;
  static constexpr float_round_style ROUND_STYLE = round_to_nearest;
  static constexpr int DIGITS = 11;
  static constexpr int DIGITS10 = 3;
  static constexpr int MAX_DIGITS10 = 5;
  static constexpr int RADIX = 2;
  static constexpr int MIN_EXPONENT = -14;
  static constexpr int MIN_EXPONENT10 = -4;
  static constexpr int MAX_EXPONENT = 15;
  static constexpr int MAX_EXPONENT10 = 4;

  static constexpr float16_t min() { return float16_t::fromBits(0x0400); }
  static constexpr float16_t lowest() { return float16_t::fromBits(0xFBFF); }
  static constexpr float16_t max() { return float16_t::fromBits(0x7BFF); }
  static constexpr float16_t infinity() { return float16_t::infinity(); }
  static constexpr float16_t quiet_NaN() { return float16_t::nan(); } // NOLINT(*-identifier-naming)
  static constexpr float16_t epsilon() { return float16_t::fromBits(0x1400); }
  static constexpr float16_t round_error() { return {0.5F}; } // NOLINT(*-identifier-naming)
};
#endif

#endif // FP16_HPP

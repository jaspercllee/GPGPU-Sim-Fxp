/* A Bison parser, made by GNU Bison 3.0.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2013 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.0.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1


/* Substitute the variable and function names.  */
#define yyparse         ptx_parse
#define yylex           ptx_lex
#define yyerror         ptx_error
#define yydebug         ptx_debug
#define yynerrs         ptx_nerrs

#define yylval          ptx_lval
#define yychar          ptx_char

/* Copy the first part of user declarations.  */

#line 75 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:339  */

# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* In a future release of Bison, this section will be replaced
   by #include "ptx.tab.h".  */
#ifndef YY_PTX_HOME_CLLI_FXP_GPGPU_SIM_DEF_BUILD_GCC_4_8_4_CUDA_7050_DEBUG_CUDA_SIM_PTX_TAB_H_INCLUDED
# define YY_PTX_HOME_CLLI_FXP_GPGPU_SIM_DEF_BUILD_GCC_4_8_4_CUDA_7050_DEBUG_CUDA_SIM_PTX_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int ptx_debug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    STRING = 258,
    OPCODE = 259,
    ALIGN_DIRECTIVE = 260,
    BRANCHTARGETS_DIRECTIVE = 261,
    BYTE_DIRECTIVE = 262,
    CALLPROTOTYPE_DIRECTIVE = 263,
    CALLTARGETS_DIRECTIVE = 264,
    CONST_DIRECTIVE = 265,
    CONSTPTR_DIRECTIVE = 266,
    PTR_DIRECTIVE = 267,
    ENTRY_DIRECTIVE = 268,
    EXTERN_DIRECTIVE = 269,
    WEAK_DIRECTIVE = 270,
    FILE_DIRECTIVE = 271,
    FUNC_DIRECTIVE = 272,
    GLOBAL_DIRECTIVE = 273,
    LOCAL_DIRECTIVE = 274,
    LOC_DIRECTIVE = 275,
    MAXNCTAPERSM_DIRECTIVE = 276,
    MAXNNREG_DIRECTIVE = 277,
    MAXNTID_DIRECTIVE = 278,
    MINNCTAPERSM_DIRECTIVE = 279,
    PARAM_DIRECTIVE = 280,
    PRAGMA_DIRECTIVE = 281,
    REG_DIRECTIVE = 282,
    REQNTID_DIRECTIVE = 283,
    SECTION_DIRECTIVE = 284,
    SHARED_DIRECTIVE = 285,
    SREG_DIRECTIVE = 286,
    STRUCT_DIRECTIVE = 287,
    SURF_DIRECTIVE = 288,
    TARGET_DIRECTIVE = 289,
    TEX_DIRECTIVE = 290,
    UNION_DIRECTIVE = 291,
    VERSION_DIRECTIVE = 292,
    ADDRESS_SIZE_DIRECTIVE = 293,
    VISIBLE_DIRECTIVE = 294,
    IDENTIFIER = 295,
    INT_OPERAND = 296,
    FLOAT_OPERAND = 297,
    DOUBLE_OPERAND = 298,
    S8_TYPE = 299,
    S16_TYPE = 300,
    S32_TYPE = 301,
    S64_TYPE = 302,
    U8_TYPE = 303,
    U16_TYPE = 304,
    U32_TYPE = 305,
    U64_TYPE = 306,
    F16_TYPE = 307,
    F32_TYPE = 308,
    F64_TYPE = 309,
    FF64_TYPE = 310,
    B8_TYPE = 311,
    B16_TYPE = 312,
    B32_TYPE = 313,
    B64_TYPE = 314,
    BB64_TYPE = 315,
    BB128_TYPE = 316,
    PRED_TYPE = 317,
    TEXREF_TYPE = 318,
    SAMPLERREF_TYPE = 319,
    SURFREF_TYPE = 320,
    V2_TYPE = 321,
    V3_TYPE = 322,
    V4_TYPE = 323,
    COMMA = 324,
    PRED = 325,
    HALF_OPTION = 326,
    EXTP_OPTION = 327,
    EQ_OPTION = 328,
    NE_OPTION = 329,
    LT_OPTION = 330,
    LE_OPTION = 331,
    GT_OPTION = 332,
    GE_OPTION = 333,
    LO_OPTION = 334,
    LS_OPTION = 335,
    HI_OPTION = 336,
    HS_OPTION = 337,
    EQU_OPTION = 338,
    NEU_OPTION = 339,
    LTU_OPTION = 340,
    LEU_OPTION = 341,
    GTU_OPTION = 342,
    GEU_OPTION = 343,
    NUM_OPTION = 344,
    NAN_OPTION = 345,
    CF_OPTION = 346,
    SF_OPTION = 347,
    NSF_OPTION = 348,
    LEFT_SQUARE_BRACKET = 349,
    RIGHT_SQUARE_BRACKET = 350,
    WIDE_OPTION = 351,
    SPECIAL_REGISTER = 352,
    MINUS = 353,
    PLUS = 354,
    COLON = 355,
    SEMI_COLON = 356,
    EXCLAMATION = 357,
    PIPE = 358,
    RIGHT_BRACE = 359,
    LEFT_BRACE = 360,
    EQUALS = 361,
    PERIOD = 362,
    BACKSLASH = 363,
    DIMENSION_MODIFIER = 364,
    RN_OPTION = 365,
    RZ_OPTION = 366,
    RM_OPTION = 367,
    RP_OPTION = 368,
    RNI_OPTION = 369,
    RZI_OPTION = 370,
    RMI_OPTION = 371,
    RPI_OPTION = 372,
    UNI_OPTION = 373,
    GEOM_MODIFIER_1D = 374,
    GEOM_MODIFIER_2D = 375,
    GEOM_MODIFIER_3D = 376,
    SAT_OPTION = 377,
    FTZ_OPTION = 378,
    NEG_OPTION = 379,
    SYNC_OPTION = 380,
    RED_OPTION = 381,
    ARRIVE_OPTION = 382,
    ATOMIC_POPC = 383,
    ATOMIC_AND = 384,
    ATOMIC_OR = 385,
    ATOMIC_XOR = 386,
    ATOMIC_CAS = 387,
    ATOMIC_EXCH = 388,
    ATOMIC_ADD = 389,
    ATOMIC_INC = 390,
    ATOMIC_DEC = 391,
    ATOMIC_MIN = 392,
    ATOMIC_MAX = 393,
    LEFT_ANGLE_BRACKET = 394,
    RIGHT_ANGLE_BRACKET = 395,
    LEFT_PAREN = 396,
    RIGHT_PAREN = 397,
    APPROX_OPTION = 398,
    FULL_OPTION = 399,
    ANY_OPTION = 400,
    ALL_OPTION = 401,
    BALLOT_OPTION = 402,
    GLOBAL_OPTION = 403,
    CTA_OPTION = 404,
    SYS_OPTION = 405,
    EXIT_OPTION = 406,
    ABS_OPTION = 407,
    TO_OPTION = 408,
    CA_OPTION = 409,
    CG_OPTION = 410,
    CS_OPTION = 411,
    LU_OPTION = 412,
    CV_OPTION = 413,
    WB_OPTION = 414,
    WT_OPTION = 415,
    NC_OPTION = 416,
    UP_OPTION = 417,
    DOWN_OPTION = 418,
    BFLY_OPTION = 419,
    IDX_OPTION = 420
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE YYSTYPE;
union YYSTYPE
{
#line 30 "ptx.y" /* yacc.c:355  */

  double double_value;
  float  float_value;
  int    int_value;
  char * string_value;
  void * ptr_value;

#line 289 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:355  */
};
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE ptx_lval;

int ptx_parse (void);

#endif /* !YY_PTX_HOME_CLLI_FXP_GPGPU_SIM_DEF_BUILD_GCC_4_8_4_CUDA_7050_DEBUG_CUDA_SIM_PTX_TAB_H_INCLUDED  */

/* Copy the second part of user declarations.  */
#line 206 "ptx.y" /* yacc.c:358  */

  	#include "ptx_parser.h"
	#include <stdlib.h>
	#include <string.h>
	#include <math.h>
	void syntax_not_implemented();
	extern int g_func_decl;
	int ptx_lex(void);
	int ptx_error(const char *);

#line 314 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:358  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn YY_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif


#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  2
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   590

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  166
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  66
/* YYNRULES -- Number of rules.  */
#define YYNRULES  286
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  395

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   420

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     155,   156,   157,   158,   159,   160,   161,   162,   163,   164,
     165
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   219,   219,   220,   221,   222,   225,   225,   226,   226,
     226,   229,   232,   233,   236,   237,   240,   240,   240,   241,
     241,   242,   245,   245,   245,   246,   249,   250,   251,   252,
     253,   254,   255,   256,   259,   260,   261,   261,   263,   263,
     264,   264,   266,   267,   268,   270,   271,   272,   274,   276,
     278,   279,   280,   281,   282,   282,   283,   283,   286,   287,
     288,   289,   290,   291,   292,   293,   294,   295,   296,   297,
     300,   301,   302,   303,   306,   308,   309,   311,   312,   324,
     325,   328,   329,   331,   332,   333,   334,   335,   338,   340,
     341,   342,   345,   346,   347,   348,   349,   350,   351,   354,
     355,   358,   359,   360,   363,   364,   365,   366,   367,   368,
     369,   370,   371,   372,   373,   374,   375,   376,   377,   378,
     379,   380,   381,   382,   383,   384,   387,   388,   390,   391,
     393,   394,   395,   397,   397,   398,   399,   400,   401,   404,
     404,   405,   407,   408,   409,   410,   411,   412,   413,   414,
     415,   416,   417,   418,   419,   422,   423,   425,   426,   427,
     428,   429,   430,   431,   432,   433,   434,   435,   436,   437,
     438,   439,   440,   441,   442,   443,   444,   445,   446,   447,
     448,   449,   450,   451,   452,   453,   454,   455,   456,   457,
     458,   459,   460,   461,   462,   463,   464,   465,   468,   469,
     470,   471,   472,   473,   474,   475,   476,   477,   478,   481,
     482,   484,   485,   486,   487,   490,   491,   492,   493,   496,
     497,   498,   499,   500,   501,   502,   503,   504,   505,   506,
     507,   508,   509,   510,   511,   512,   513,   516,   517,   519,
     520,   521,   522,   523,   524,   525,   526,   527,   528,   529,
     530,   531,   532,   533,   534,   535,   536,   537,   538,   541,
     542,   543,   544,   547,   547,   552,   553,   556,   557,   558,
     559,   560,   563,   564,   565,   566,   567,   568,   569,   572,
     573,   574,   577,   578,   579,   580,   581
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "STRING", "OPCODE", "ALIGN_DIRECTIVE",
  "BRANCHTARGETS_DIRECTIVE", "BYTE_DIRECTIVE", "CALLPROTOTYPE_DIRECTIVE",
  "CALLTARGETS_DIRECTIVE", "CONST_DIRECTIVE", "CONSTPTR_DIRECTIVE",
  "PTR_DIRECTIVE", "ENTRY_DIRECTIVE", "EXTERN_DIRECTIVE", "WEAK_DIRECTIVE",
  "FILE_DIRECTIVE", "FUNC_DIRECTIVE", "GLOBAL_DIRECTIVE",
  "LOCAL_DIRECTIVE", "LOC_DIRECTIVE", "MAXNCTAPERSM_DIRECTIVE",
  "MAXNNREG_DIRECTIVE", "MAXNTID_DIRECTIVE", "MINNCTAPERSM_DIRECTIVE",
  "PARAM_DIRECTIVE", "PRAGMA_DIRECTIVE", "REG_DIRECTIVE",
  "REQNTID_DIRECTIVE", "SECTION_DIRECTIVE", "SHARED_DIRECTIVE",
  "SREG_DIRECTIVE", "STRUCT_DIRECTIVE", "SURF_DIRECTIVE",
  "TARGET_DIRECTIVE", "TEX_DIRECTIVE", "UNION_DIRECTIVE",
  "VERSION_DIRECTIVE", "ADDRESS_SIZE_DIRECTIVE", "VISIBLE_DIRECTIVE",
  "IDENTIFIER", "INT_OPERAND", "FLOAT_OPERAND", "DOUBLE_OPERAND",
  "S8_TYPE", "S16_TYPE", "S32_TYPE", "S64_TYPE", "U8_TYPE", "U16_TYPE",
  "U32_TYPE", "U64_TYPE", "F16_TYPE", "F32_TYPE", "F64_TYPE", "FF64_TYPE",
  "B8_TYPE", "B16_TYPE", "B32_TYPE", "B64_TYPE", "BB64_TYPE", "BB128_TYPE",
  "PRED_TYPE", "TEXREF_TYPE", "SAMPLERREF_TYPE", "SURFREF_TYPE", "V2_TYPE",
  "V3_TYPE", "V4_TYPE", "COMMA", "PRED", "HALF_OPTION", "EXTP_OPTION",
  "EQ_OPTION", "NE_OPTION", "LT_OPTION", "LE_OPTION", "GT_OPTION",
  "GE_OPTION", "LO_OPTION", "LS_OPTION", "HI_OPTION", "HS_OPTION",
  "EQU_OPTION", "NEU_OPTION", "LTU_OPTION", "LEU_OPTION", "GTU_OPTION",
  "GEU_OPTION", "NUM_OPTION", "NAN_OPTION", "CF_OPTION", "SF_OPTION",
  "NSF_OPTION", "LEFT_SQUARE_BRACKET", "RIGHT_SQUARE_BRACKET",
  "WIDE_OPTION", "SPECIAL_REGISTER", "MINUS", "PLUS", "COLON",
  "SEMI_COLON", "EXCLAMATION", "PIPE", "RIGHT_BRACE", "LEFT_BRACE",
  "EQUALS", "PERIOD", "BACKSLASH", "DIMENSION_MODIFIER", "RN_OPTION",
  "RZ_OPTION", "RM_OPTION", "RP_OPTION", "RNI_OPTION", "RZI_OPTION",
  "RMI_OPTION", "RPI_OPTION", "UNI_OPTION", "GEOM_MODIFIER_1D",
  "GEOM_MODIFIER_2D", "GEOM_MODIFIER_3D", "SAT_OPTION", "FTZ_OPTION",
  "NEG_OPTION", "SYNC_OPTION", "RED_OPTION", "ARRIVE_OPTION",
  "ATOMIC_POPC", "ATOMIC_AND", "ATOMIC_OR", "ATOMIC_XOR", "ATOMIC_CAS",
  "ATOMIC_EXCH", "ATOMIC_ADD", "ATOMIC_INC", "ATOMIC_DEC", "ATOMIC_MIN",
  "ATOMIC_MAX", "LEFT_ANGLE_BRACKET", "RIGHT_ANGLE_BRACKET", "LEFT_PAREN",
  "RIGHT_PAREN", "APPROX_OPTION", "FULL_OPTION", "ANY_OPTION",
  "ALL_OPTION", "BALLOT_OPTION", "GLOBAL_OPTION", "CTA_OPTION",
  "SYS_OPTION", "EXIT_OPTION", "ABS_OPTION", "TO_OPTION", "CA_OPTION",
  "CG_OPTION", "CS_OPTION", "LU_OPTION", "CV_OPTION", "WB_OPTION",
  "WT_OPTION", "NC_OPTION", "UP_OPTION", "DOWN_OPTION", "BFLY_OPTION",
  "IDX_OPTION", "$accept", "input", "function_defn", "$@1", "$@2", "$@3",
  "block_spec", "block_spec_list", "function_decl", "$@4", "$@5", "$@6",
  "function_ident_param", "$@7", "$@8", "function_decl_header",
  "param_list", "$@9", "param_entry", "$@10", "$@11", "ptr_spec",
  "ptr_space_spec", "ptr_align_spec", "statement_block", "statement_list",
  "$@12", "$@13", "directive_statement", "variable_declaration",
  "variable_spec", "identifier_list", "identifier_spec", "var_spec_list",
  "var_spec", "align_spec", "space_spec", "addressable_spec", "type_spec",
  "vector_spec", "scalar_type", "initializer_list", "literal_list",
  "instruction_statement", "instruction", "$@14", "opcode_spec", "$@15",
  "pred_spec", "option_list", "option", "atomic_operation_spec",
  "rounding_mode", "floating_point_rounding_mode", "integer_rounding_mode",
  "compare_spec", "operand_list", "operand", "vector_operand",
  "tex_operand", "$@16", "builtin_operand", "memory_operand",
  "twin_operand", "literal_operand", "address_expression", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372,   373,   374,
     375,   376,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405,   406,   407,   408,   409,   410,   411,   412,   413,   414,
     415,   416,   417,   418,   419,   420
};
# endif

#define YYPACT_NINF -287

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-287)))

#define YYTABLE_NINF -142

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
    -287,   350,  -287,   -31,  -287,   -25,  -287,     7,    12,   -11,
    -287,  -287,  -287,    -1,  -287,   103,  -287,  -287,  -287,  -287,
     129,  -287,   144,   165,    85,  -287,  -287,  -287,  -287,  -287,
    -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,
    -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,
    -287,   -12,   -34,  -287,    80,   171,   479,  -287,  -287,  -287,
    -287,  -287,   504,  -287,  -287,   147,  -287,  -287,  -287,   217,
     182,   135,   170,   150,  -287,  -287,  -287,  -287,   148,    76,
    -287,   221,  -287,     1,   195,   196,  -287,  -287,  -287,  -287,
     230,   241,   273,  -287,   280,  -287,   415,  -287,   306,   308,
     312,  -287,    76,    78,   213,  -287,   -27,   315,   171,    -4,
     288,   317,  -287,   290,   130,   262,    -6,   270,   278,   148,
    -287,  -287,   271,   143,   369,  -287,   305,  -287,   148,  -287,
    -287,  -287,   236,   245,   284,  -287,   250,  -287,  -287,  -287,
    -287,    -4,  -287,  -287,   351,   322,   353,    -2,  -287,   497,
     381,  -287,   148,  -287,  -287,  -287,  -287,   178,   136,   313,
     164,   383,   384,   158,  -287,   358,  -287,  -287,  -287,  -287,
    -287,   335,   396,  -287,   479,   479,  -287,  -287,  -287,  -287,
     334,    35,  -287,  -287,   398,  -287,  -287,  -287,  -287,  -287,
    -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,
    -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,
    -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,
    -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,
    -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,
    -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,
    -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,
    -287,  -287,    -2,  -287,  -287,  -287,  -287,  -287,  -287,  -287,
    -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,  -287,
    -287,  -287,  -287,   233,   402,   404,   407,   126,  -287,   356,
    -287,   186,   238,   140,  -287,  -287,  -287,    99,   314,   149,
    -287,   388,   446,   171,   221,    78,  -287,   258,  -287,  -287,
    -287,   115,  -287,   391,   392,   393,  -287,   138,   163,  -287,
    -287,  -287,   449,  -287,  -287,  -287,   134,   397,   452,  -287,
    -287,   124,  -287,   426,   455,     2,   171,  -287,  -287,   -51,
    -287,  -287,   -14,  -287,  -287,  -287,  -287,  -287,  -287,  -287,
     394,  -287,   105,   431,  -287,   359,   158,  -287,   461,  -287,
    -287,  -287,   498,  -287,  -287,  -287,  -287,   206,   266,   410,
     467,  -287,   158,  -287,  -287,  -287,    78,  -287,  -287,   240,
    -287,  -287,   106,   439,  -287,  -287,  -287,   471,  -287,   372,
     411,   158,  -287,   374,  -287
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       2,     0,     1,     0,    92,     0,    26,    86,    87,     0,
      29,    93,    94,     0,    95,     0,    89,    96,    90,    97,
       0,    98,     0,     0,     0,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124,   125,   101,   102,   103,
       4,     5,    21,     3,     0,     0,    74,    81,    85,    83,
      91,    84,     0,    99,    88,     0,    32,    28,    31,     0,
       0,     0,    64,    59,    61,    27,    30,    69,     0,     0,
      16,     0,    58,    77,    70,    75,    86,    87,    82,   100,
       0,    65,     0,    68,     0,    60,    56,     7,     0,     0,
       0,    14,     9,     0,    25,    20,     0,     0,     0,     0,
       0,     0,    67,    62,   139,     0,     0,     0,    54,     0,
      50,    51,     0,   138,     0,    13,     0,    12,     0,    15,
      38,    40,     0,     0,     0,    79,     0,    76,   279,   280,
     281,     0,    71,    72,     0,     0,     0,     0,   131,   142,
       0,    49,     0,    52,    53,    57,   130,   239,     0,   266,
       0,     0,     0,     0,   137,   237,   245,   247,   244,   242,
     243,     0,     0,    10,     0,     0,    17,    23,    80,    78,
       0,     0,   128,    73,     0,    63,   184,   185,   219,   220,
     221,   222,   223,   224,   225,   226,   227,   228,   229,   230,
     231,   232,   233,   234,   235,   236,   165,   211,   212,   213,
     214,   215,   216,   217,   218,   164,   172,   173,   174,   175,
     176,   177,   161,   163,   162,   199,   198,   200,   201,   202,
     203,   204,   205,   206,   207,   208,   178,   179,   166,   167,
     168,   169,   170,   171,   180,   181,   183,   186,   187,   188,
     189,   190,   191,   192,   193,   194,   195,   196,   197,   159,
     157,   140,   155,   182,   160,   209,   210,   158,   145,   147,
     144,   146,   148,   149,   151,   150,   152,   153,   154,   143,
      55,   249,   251,     0,     0,     0,     0,   282,   286,     0,
     265,   241,     0,     0,   246,   271,   240,     0,     0,     0,
     132,     0,    42,     0,     0,    34,   127,     0,   126,    66,
     156,   282,   279,     0,     0,     0,   248,   253,   256,   263,
     283,   284,     0,   267,   250,   252,   282,     0,     0,   262,
     133,     0,   238,   237,     0,     0,     0,    41,    18,     0,
      35,   129,     0,   270,   269,   268,   254,   255,   257,   258,
       0,   285,     0,     0,   136,     0,     0,    11,     0,    45,
      46,    47,     0,    44,    39,    36,    24,   272,     0,     0,
       0,   259,     0,   135,    48,    43,     0,   273,   274,   275,
     278,   264,     0,     0,    37,   276,   277,     0,   260,     0,
       0,     0,   261,     0,   134
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -287,  -287,  -287,  -287,  -287,  -287,   416,  -287,   516,  -287,
    -287,  -287,   215,  -287,  -287,  -287,  -287,  -287,  -286,  -287,
    -287,  -287,  -287,   159,    60,  -287,  -287,  -287,   -83,  -287,
     177,  -287,  -106,  -287,   464,  -287,  -287,  -125,   -80,  -287,
     460,   433,  -287,   458,   453,  -287,  -287,  -287,  -287,   316,
    -287,  -287,  -287,  -287,  -287,  -287,  -123,  -122,  -157,  -287,
    -287,  -287,  -155,  -287,  -105,   296
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,    50,    78,    79,   128,   101,   102,   117,   103,
     304,    81,   105,   133,   305,    52,   339,   376,   132,   174,
     175,   336,   362,   363,    97,   118,   152,   119,    53,    54,
      55,    84,    85,    56,    57,    58,    59,    60,    61,    62,
      63,   142,   181,   121,   122,   353,   123,   147,   124,   261,
     262,   263,   264,   265,   266,   267,   332,   333,   166,   167,
     350,   168,   169,   313,   170,   289
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
     164,   165,   137,   294,   143,   295,   -19,   358,     4,    -8,
      64,    -8,    -8,   120,   134,    65,    11,    12,   365,   340,
     359,   360,   259,    14,    66,    67,   367,   351,    17,    68,
      69,    19,   361,    21,   149,   153,   182,   138,   139,   140,
      70,   298,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,   260,   135,   186,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     197,   198,   199,   200,   201,   202,   203,   204,   205,    77,
     384,   366,   368,    -6,   206,   106,   150,    98,    75,    99,
     100,   141,    76,   130,   307,   131,    71,    80,   207,   208,
     209,   210,   211,   212,   213,   214,   215,   216,   217,   218,
     219,   220,   221,   222,   223,   224,   225,   226,   227,   228,
     229,   230,   231,   232,   233,   234,   235,   259,   295,   308,
     107,   236,   237,   238,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   157,   138,   139,   140,   328,    72,
    -141,  -141,  -141,  -141,   370,   387,   287,   288,   314,   155,
     327,    82,   260,   157,   138,   139,   140,    73,   173,   157,
     138,   139,   140,   369,   320,   319,   321,   337,   157,   138,
     139,   140,   341,   329,   291,   320,    74,   321,   355,   371,
     388,    83,   280,   320,   342,   321,    90,   346,   158,   347,
      91,   159,   160,    92,  -141,   322,   161,  -141,  -141,   162,
     364,  -141,  -141,   322,   292,  -141,    93,   158,   293,    94,
     159,   160,   348,   158,   349,   161,   159,   160,   162,    95,
     383,   161,   158,    96,   162,   159,   160,   281,   292,   282,
     161,   104,   293,   162,   108,   324,   354,   325,   393,   162,
     110,  -141,   283,   311,   312,   139,   140,   284,   326,   288,
     283,   285,   114,     3,   163,   377,   286,   378,     4,     5,
     331,     6,     7,     8,     9,    10,    11,    12,    13,   138,
     139,   140,   109,    14,    15,    16,   379,   380,    17,    18,
     111,    19,    20,    21,   112,    22,    23,    24,   115,   385,
     113,   386,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,   125,   116,   126,
       2,   302,   303,   127,   -22,     3,   136,   144,   145,   146,
       4,     5,   148,     6,     7,     8,     9,    10,    11,    12,
      13,    77,   156,   114,   172,    14,    15,    16,   176,   178,
      17,    18,   151,    19,    20,    21,   177,    22,    23,    24,
     179,   184,   183,   185,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,   114,
       3,   279,   290,   296,   297,     4,     5,   299,     6,     7,
       8,     9,    10,    11,    12,    13,   300,   301,   306,   309,
      14,    15,    16,   316,   317,    17,    18,   318,    19,    20,
      21,   323,    22,    23,    24,   115,   330,   334,   335,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,     3,   116,   343,   344,   345,     4,
     351,   283,   352,    86,    87,   356,   357,    11,    12,   162,
     372,   373,   374,   358,    14,   381,    16,   382,   389,    17,
      18,   390,    19,   391,    21,   392,   394,    51,   129,   338,
      88,   375,    89,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
     268,   269,   270,   271,   180,   272,   154,   171,   310,   315,
     273,   274,     0,     0,   275,     0,     0,     0,   276,   277,
     278
};

static const yytype_int16 yycheck[] =
{
     123,   123,   108,   160,   109,   160,    40,     5,    10,    21,
      41,    23,    24,    96,    41,    40,    18,    19,    69,   305,
      18,    19,   147,    25,    17,    13,    40,    41,    30,    17,
      41,    33,    30,    35,    40,   118,   141,    41,    42,    43,
      41,   163,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,   147,    95,    71,
      72,    73,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,   101,
     376,   142,   106,   105,    96,    94,   102,    21,    13,    23,
      24,   105,    17,    25,    69,    27,     3,   141,   110,   111,
     112,   113,   114,   115,   116,   117,   118,   119,   120,   121,
     122,   123,   124,   125,   126,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,   262,   293,   104,
     139,   143,   144,   145,   146,   147,   148,   149,   150,   151,
     152,   153,   154,   155,   156,   157,   158,   159,   160,   161,
     162,   163,   164,   165,    40,    41,    42,    43,    69,    40,
      40,    41,    42,    43,    69,    69,    40,    41,   283,   119,
      40,   101,   262,    40,    41,    42,    43,    43,   128,    40,
      41,    42,    43,   350,    79,    69,    81,   303,    40,    41,
      42,    43,   307,   104,    40,    79,    41,    81,   331,   104,
     104,    40,   152,    79,    99,    81,    69,    79,    94,    81,
       3,    97,    98,    41,    94,    99,   102,    97,    98,   105,
     336,   101,   102,    99,    94,   105,   101,    94,    98,    69,
      97,    98,    79,    94,    81,   102,    97,    98,   105,    99,
     372,   102,    94,   105,   105,    97,    98,    79,    94,    81,
     102,    40,    98,   105,    69,    79,   142,    81,   391,   105,
      40,   141,    94,    40,    41,    42,    43,    99,    40,    41,
      94,   103,     4,     5,   141,    79,   108,    81,    10,    11,
     141,    13,    14,    15,    16,    17,    18,    19,    20,    41,
      42,    43,   106,    25,    26,    27,    40,    41,    30,    31,
      69,    33,    34,    35,    41,    37,    38,    39,    40,    79,
      40,    81,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    41,    70,    41,
       0,   174,   175,    41,   141,     5,    41,    69,    41,    69,
      10,    11,   100,    13,    14,    15,    16,    17,    18,    19,
      20,   101,   101,     4,    69,    25,    26,    27,   142,    95,
      30,    31,   104,    33,    34,    35,   141,    37,    38,    39,
     140,    69,    41,    40,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    57,    58,    59,
      60,    61,    62,    63,    64,    65,    66,    67,    68,     4,
       5,    40,   109,    40,    40,    10,    11,    69,    13,    14,
      15,    16,    17,    18,    19,    20,   101,    41,   104,    41,
      25,    26,    27,    41,    40,    30,    31,    40,    33,    34,
      35,    95,    37,    38,    39,    40,   142,    69,    12,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,     5,    70,    95,    95,    95,    10,
      41,    94,    40,    14,    15,    69,    41,    18,    19,   105,
      69,   142,    41,     5,    25,    95,    27,    40,    69,    30,
      31,    40,    33,   141,    35,   104,   142,     1,   102,   304,
      56,   362,    62,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
      73,    74,    75,    76,   141,    78,   118,   124,   262,   283,
      83,    84,    -1,    -1,    87,    -1,    -1,    -1,    91,    92,
      93
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,   167,     0,     5,    10,    11,    13,    14,    15,    16,
      17,    18,    19,    20,    25,    26,    27,    30,    31,    33,
      34,    35,    37,    38,    39,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
     168,   174,   181,   194,   195,   196,   199,   200,   201,   202,
     203,   204,   205,   206,    41,    40,    17,    13,    17,    41,
      41,     3,    40,    43,    41,    13,    17,   101,   169,   170,
     141,   177,   101,    40,   197,   198,    14,    15,   200,   206,
      69,     3,    41,   101,    69,    99,   105,   190,    21,    23,
      24,   172,   173,   175,    40,   178,    94,   139,    69,   106,
      40,    69,    41,    40,     4,    40,    70,   174,   191,   193,
     194,   209,   210,   212,   214,    41,    41,    41,   171,   172,
      25,    27,   184,   179,    41,    95,    41,   198,    41,    42,
      43,   105,   207,   230,    69,    41,    69,   213,   100,    40,
     102,   104,   192,   194,   209,   190,   101,    40,    94,    97,
      98,   102,   105,   141,   222,   223,   224,   225,   227,   228,
     230,   210,    69,   190,   185,   186,   142,   141,    95,   140,
     207,   208,   230,    41,    69,    40,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    96,   110,   111,   112,
     113,   114,   115,   116,   117,   118,   119,   120,   121,   122,
     123,   124,   125,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,   137,   138,   143,   144,   145,   146,
     147,   148,   149,   150,   151,   152,   153,   154,   155,   156,
     157,   158,   159,   160,   161,   162,   163,   164,   165,   203,
     204,   215,   216,   217,   218,   219,   220,   221,    73,    74,
      75,    76,    78,    83,    84,    87,    91,    92,    93,    40,
     190,    79,    81,    94,    99,   103,   108,    40,    41,   231,
     109,    40,    94,    98,   224,   228,    40,    40,   223,    69,
     101,    41,   196,   196,   176,   180,   104,    69,   104,    41,
     215,    40,    41,   229,   230,   231,    41,    40,    40,    69,
      79,    81,    99,    95,    79,    81,    40,    40,    69,   104,
     142,   141,   222,   223,    69,    12,   187,   198,   178,   182,
     184,   230,    99,    95,    95,    95,    79,    81,    79,    81,
     226,    41,    40,   211,   142,   222,    69,    41,     5,    18,
      19,    30,   188,   189,   198,    69,   142,    40,   106,   224,
      69,   104,    69,   142,    41,   189,   183,    79,    81,    40,
      41,    95,    40,   223,   184,    79,    81,    69,   104,    69,
      40,   141,   104,   222,   142
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,   166,   167,   167,   167,   167,   169,   168,   170,   171,
     168,   172,   172,   172,   173,   173,   175,   176,   174,   177,
     174,   174,   179,   180,   178,   178,   181,   181,   181,   181,
     181,   181,   181,   181,   182,   182,   183,   182,   185,   184,
     186,   184,   187,   187,   187,   188,   188,   188,   189,   190,
     191,   191,   191,   191,   192,   191,   193,   191,   194,   194,
     194,   194,   194,   194,   194,   194,   194,   194,   194,   194,
     195,   195,   195,   195,   196,   197,   197,   198,   198,   198,
     198,   199,   199,   200,   200,   200,   200,   200,   201,   202,
     202,   202,   203,   203,   203,   203,   203,   203,   203,   204,
     204,   205,   205,   205,   206,   206,   206,   206,   206,   206,
     206,   206,   206,   206,   206,   206,   206,   206,   206,   206,
     206,   206,   206,   206,   206,   206,   207,   207,   208,   208,
     209,   209,   209,   211,   210,   210,   210,   210,   210,   213,
     212,   212,   214,   214,   214,   214,   214,   214,   214,   214,
     214,   214,   214,   214,   214,   215,   215,   216,   216,   216,
     216,   216,   216,   216,   216,   216,   216,   216,   216,   216,
     216,   216,   216,   216,   216,   216,   216,   216,   216,   216,
     216,   216,   216,   216,   216,   216,   216,   216,   216,   216,
     216,   216,   216,   216,   216,   216,   216,   216,   217,   217,
     217,   217,   217,   217,   217,   217,   217,   217,   217,   218,
     218,   219,   219,   219,   219,   220,   220,   220,   220,   221,
     221,   221,   221,   221,   221,   221,   221,   221,   221,   221,
     221,   221,   221,   221,   221,   221,   221,   222,   222,   223,
     223,   223,   223,   223,   223,   223,   223,   223,   223,   223,
     223,   223,   223,   223,   223,   223,   223,   223,   223,   224,
     224,   224,   224,   226,   225,   227,   227,   228,   228,   228,
     228,   228,   229,   229,   229,   229,   229,   229,   229,   230,
     230,   230,   231,   231,   231,   231,   231
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     2,     2,     2,     0,     3,     0,     0,
       5,     6,     2,     2,     1,     2,     0,     0,     7,     0,
       3,     1,     0,     0,     6,     1,     1,     2,     2,     1,
       2,     2,     2,     2,     0,     1,     0,     4,     0,     5,
       0,     4,     0,     3,     2,     1,     1,     1,     2,     3,
       1,     1,     2,     2,     0,     3,     0,     2,     2,     2,
       3,     2,     4,     6,     2,     3,     7,     4,     3,     2,
       2,     4,     4,     6,     1,     1,     3,     1,     4,     3,
       4,     1,     2,     1,     1,     1,     1,     1,     2,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       2,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     3,     3,     1,     3,
       2,     2,     3,     0,    11,     6,     5,     2,     1,     0,
       3,     1,     2,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     1,     2,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     3,     1,
       2,     2,     1,     1,     1,     1,     2,     1,     3,     2,
       3,     2,     3,     3,     4,     4,     3,     4,     4,     5,
       7,     9,     3,     0,     6,     2,     1,     3,     4,     4,
       4,     2,     3,     4,     4,     4,     5,     5,     4,     1,
       1,     1,     1,     2,     2,     3,     1
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256



/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  YYUSE (yytype);
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, int yyrule)
{
  unsigned long int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &(yyvsp[(yyi + 1) - (yynrhs)])
                                              );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen (const char *yystr)
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
{
  YYUSE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        YYSTYPE *yyvs1 = yyvs;
        yytype_int16 *yyss1 = yyss;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yystacksize);

        yyss = yyss1;
        yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yytype_int16 *yyss1 = yyss;
        union yyalloc *yyptr =
          (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 6:
#line 225 "ptx.y" /* yacc.c:1646  */
    { set_symtab((yyvsp[0].ptr_value)); func_header(".skip"); }
#line 1794 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 225 "ptx.y" /* yacc.c:1646  */
    { end_function(); }
#line 1800 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 226 "ptx.y" /* yacc.c:1646  */
    { set_symtab((yyvsp[0].ptr_value)); }
#line 1806 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 226 "ptx.y" /* yacc.c:1646  */
    { func_header(".skip"); }
#line 1812 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 226 "ptx.y" /* yacc.c:1646  */
    { end_function(); }
#line 1818 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 229 "ptx.y" /* yacc.c:1646  */
    {func_header_info_int(".maxntid", (yyvsp[-4].int_value));
										func_header_info_int(",", (yyvsp[-2].int_value));
										func_header_info_int(",", (yyvsp[0].int_value)); }
#line 1826 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 232 "ptx.y" /* yacc.c:1646  */
    { func_header_info_int(".minnctapersm", (yyvsp[0].int_value)); printf("GPGPU-Sim: Warning: .minnctapersm ignored. \n"); }
#line 1832 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 233 "ptx.y" /* yacc.c:1646  */
    { func_header_info_int(".maxnctapersm", (yyvsp[0].int_value)); printf("GPGPU-Sim: Warning: .maxnctapersm ignored. \n"); }
#line 1838 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 240 "ptx.y" /* yacc.c:1646  */
    { start_function((yyvsp[-1].int_value)); func_header_info("(");}
#line 1844 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 240 "ptx.y" /* yacc.c:1646  */
    {func_header_info(")");}
#line 1850 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 240 "ptx.y" /* yacc.c:1646  */
    { (yyval.ptr_value) = reset_symtab(); }
#line 1856 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 241 "ptx.y" /* yacc.c:1646  */
    { start_function((yyvsp[0].int_value)); }
#line 1862 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 241 "ptx.y" /* yacc.c:1646  */
    { (yyval.ptr_value) = reset_symtab(); }
#line 1868 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 242 "ptx.y" /* yacc.c:1646  */
    { start_function((yyvsp[0].int_value)); add_function_name(""); g_func_decl=0; (yyval.ptr_value) = reset_symtab(); }
#line 1874 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 245 "ptx.y" /* yacc.c:1646  */
    { add_function_name((yyvsp[0].string_value)); }
#line 1880 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 245 "ptx.y" /* yacc.c:1646  */
    {func_header_info("(");}
#line 1886 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 245 "ptx.y" /* yacc.c:1646  */
    { g_func_decl=0; func_header_info(")"); }
#line 1892 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 246 "ptx.y" /* yacc.c:1646  */
    { add_function_name((yyvsp[0].string_value)); g_func_decl=0; }
#line 1898 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 249 "ptx.y" /* yacc.c:1646  */
    { (yyval.int_value) = 1; g_func_decl=1; func_header(".entry"); }
#line 1904 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 250 "ptx.y" /* yacc.c:1646  */
    { (yyval.int_value) = 1; g_func_decl=1; func_header(".entry"); }
#line 1910 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 251 "ptx.y" /* yacc.c:1646  */
    { (yyval.int_value) = 1; g_func_decl=1; func_header(".entry"); }
#line 1916 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 252 "ptx.y" /* yacc.c:1646  */
    { (yyval.int_value) = 0; g_func_decl=1; func_header(".func"); }
#line 1922 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 253 "ptx.y" /* yacc.c:1646  */
    { (yyval.int_value) = 0; g_func_decl=1; func_header(".func"); }
#line 1928 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 254 "ptx.y" /* yacc.c:1646  */
    { (yyval.int_value) = 0; g_func_decl=1; func_header(".func"); }
#line 1934 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 255 "ptx.y" /* yacc.c:1646  */
    { (yyval.int_value) = 2; g_func_decl=1; func_header(".func"); }
#line 1940 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 256 "ptx.y" /* yacc.c:1646  */
    { (yyval.int_value) = 0; g_func_decl=1; func_header(".func"); }
#line 1946 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 260 "ptx.y" /* yacc.c:1646  */
    { add_directive(); }
#line 1952 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 261 "ptx.y" /* yacc.c:1646  */
    {func_header_info(",");}
#line 1958 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 261 "ptx.y" /* yacc.c:1646  */
    { add_directive(); }
#line 1964 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 263 "ptx.y" /* yacc.c:1646  */
    { add_space_spec(param_space_unclassified,0); }
#line 1970 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 263 "ptx.y" /* yacc.c:1646  */
    { add_function_arg(); }
#line 1976 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 264 "ptx.y" /* yacc.c:1646  */
    { add_space_spec(reg_space,0); }
#line 1982 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 264 "ptx.y" /* yacc.c:1646  */
    { add_function_arg(); }
#line 1988 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 270 "ptx.y" /* yacc.c:1646  */
    { add_ptr_spec(global_space); }
#line 1994 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 271 "ptx.y" /* yacc.c:1646  */
    { add_ptr_spec(local_space); }
#line 2000 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 272 "ptx.y" /* yacc.c:1646  */
    { add_ptr_spec(shared_space); }
#line 2006 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 278 "ptx.y" /* yacc.c:1646  */
    { add_directive(); }
#line 2012 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 279 "ptx.y" /* yacc.c:1646  */
    { add_instruction(); }
#line 2018 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 280 "ptx.y" /* yacc.c:1646  */
    { add_directive(); }
#line 2024 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 281 "ptx.y" /* yacc.c:1646  */
    { add_instruction(); }
#line 2030 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 282 "ptx.y" /* yacc.c:1646  */
    {start_inst_group();}
#line 2036 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 282 "ptx.y" /* yacc.c:1646  */
    {end_inst_group();}
#line 2042 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 283 "ptx.y" /* yacc.c:1646  */
    {start_inst_group();}
#line 2048 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 283 "ptx.y" /* yacc.c:1646  */
    {end_inst_group();}
#line 2054 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 287 "ptx.y" /* yacc.c:1646  */
    { add_version_info((yyvsp[0].double_value), 0); }
#line 2060 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 288 "ptx.y" /* yacc.c:1646  */
    { add_version_info((yyvsp[-1].double_value),1); }
#line 2066 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 289 "ptx.y" /* yacc.c:1646  */
    {/*Do nothing*/}
#line 2072 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 290 "ptx.y" /* yacc.c:1646  */
    { target_header2((yyvsp[-2].string_value),(yyvsp[0].string_value)); }
#line 2078 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 291 "ptx.y" /* yacc.c:1646  */
    { target_header3((yyvsp[-4].string_value),(yyvsp[-2].string_value),(yyvsp[0].string_value)); }
#line 2084 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 292 "ptx.y" /* yacc.c:1646  */
    { target_header((yyvsp[0].string_value)); }
#line 2090 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 293 "ptx.y" /* yacc.c:1646  */
    { add_file((yyvsp[-1].int_value),(yyvsp[0].string_value)); }
#line 2096 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 294 "ptx.y" /* yacc.c:1646  */
    { add_file((yyvsp[-5].int_value),(yyvsp[-4].string_value)); }
#line 2102 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 296 "ptx.y" /* yacc.c:1646  */
    { add_pragma((yyvsp[-1].string_value)); }
#line 2108 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 297 "ptx.y" /* yacc.c:1646  */
    {/*Do nothing*/}
#line 2114 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 300 "ptx.y" /* yacc.c:1646  */
    { add_variables(); }
#line 2120 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 301 "ptx.y" /* yacc.c:1646  */
    { add_variables(); }
#line 2126 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 302 "ptx.y" /* yacc.c:1646  */
    { add_variables(); }
#line 2132 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 303 "ptx.y" /* yacc.c:1646  */
    { add_constptr((yyvsp[-4].string_value), (yyvsp[-2].string_value), (yyvsp[0].int_value)); }
#line 2138 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 306 "ptx.y" /* yacc.c:1646  */
    { set_variable_type(); }
#line 2144 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 311 "ptx.y" /* yacc.c:1646  */
    { add_identifier((yyvsp[0].string_value),0,NON_ARRAY_IDENTIFIER); func_header_info((yyvsp[0].string_value));}
#line 2150 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 312 "ptx.y" /* yacc.c:1646  */
    { func_header_info((yyvsp[-3].string_value)); func_header_info_int("<", (yyvsp[-1].int_value)); func_header_info(">");
		int i,lbase,l;
		char *id = NULL;
		lbase = strlen((yyvsp[-3].string_value));
		for( i=0; i < (yyvsp[-1].int_value); i++ ) { 
			l = lbase + (int)log10(i+1)+10;
			id = (char*) malloc(l);
			snprintf(id,l,"%s%u",(yyvsp[-3].string_value),i);
			add_identifier(id,0,NON_ARRAY_IDENTIFIER); 
		}
		free((yyvsp[-3].string_value));
	}
#line 2167 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 324 "ptx.y" /* yacc.c:1646  */
    { add_identifier((yyvsp[-2].string_value),0,ARRAY_IDENTIFIER_NO_DIM); func_header_info((yyvsp[-2].string_value)); func_header_info("["); func_header_info("]");}
#line 2173 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 325 "ptx.y" /* yacc.c:1646  */
    { add_identifier((yyvsp[-3].string_value),(yyvsp[-1].int_value),ARRAY_IDENTIFIER); func_header_info((yyvsp[-3].string_value)); func_header_info_int("[",(yyvsp[-1].int_value)); func_header_info("]");}
#line 2179 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 334 "ptx.y" /* yacc.c:1646  */
    { add_extern_spec(); }
#line 2185 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 338 "ptx.y" /* yacc.c:1646  */
    { add_alignment_spec((yyvsp[0].int_value)); }
#line 2191 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 340 "ptx.y" /* yacc.c:1646  */
    {  add_space_spec(reg_space,0); }
#line 2197 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 341 "ptx.y" /* yacc.c:1646  */
    {  add_space_spec(reg_space,0); }
#line 2203 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 345 "ptx.y" /* yacc.c:1646  */
    {  add_space_spec(const_space,(yyvsp[0].int_value)); }
#line 2209 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 346 "ptx.y" /* yacc.c:1646  */
    {  add_space_spec(global_space,0); }
#line 2215 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 347 "ptx.y" /* yacc.c:1646  */
    {  add_space_spec(local_space,0); }
#line 2221 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 348 "ptx.y" /* yacc.c:1646  */
    {  add_space_spec(param_space_unclassified,0); }
#line 2227 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 349 "ptx.y" /* yacc.c:1646  */
    {  add_space_spec(shared_space,0); }
#line 2233 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 350 "ptx.y" /* yacc.c:1646  */
    {  add_space_spec(surf_space,0); }
#line 2239 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 351 "ptx.y" /* yacc.c:1646  */
    {  add_space_spec(tex_space,0); }
#line 2245 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 358 "ptx.y" /* yacc.c:1646  */
    {  add_option(V2_TYPE); func_header_info(".v2");}
#line 2251 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 359 "ptx.y" /* yacc.c:1646  */
    {  add_option(V3_TYPE); func_header_info(".v3");}
#line 2257 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 360 "ptx.y" /* yacc.c:1646  */
    {  add_option(V4_TYPE); func_header_info(".v4");}
#line 2263 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 363 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( S8_TYPE ); }
#line 2269 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 364 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( S16_TYPE ); }
#line 2275 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 365 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( S32_TYPE ); }
#line 2281 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 366 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( S64_TYPE ); }
#line 2287 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 367 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( U8_TYPE ); }
#line 2293 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 368 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( U16_TYPE ); }
#line 2299 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 369 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( U32_TYPE ); }
#line 2305 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 370 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( U64_TYPE ); }
#line 2311 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 371 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( F16_TYPE ); }
#line 2317 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 372 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( F32_TYPE ); }
#line 2323 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 373 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( F64_TYPE ); }
#line 2329 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 374 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( FF64_TYPE ); }
#line 2335 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 375 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( B8_TYPE );  }
#line 2341 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 376 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( B16_TYPE ); }
#line 2347 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 377 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( B32_TYPE ); }
#line 2353 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 378 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( B64_TYPE ); }
#line 2359 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 379 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( BB64_TYPE ); }
#line 2365 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 380 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( BB128_TYPE ); }
#line 2371 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 381 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( PRED_TYPE ); }
#line 2377 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 382 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( TEXREF_TYPE ); }
#line 2383 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 383 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( SAMPLERREF_TYPE ); }
#line 2389 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 384 "ptx.y" /* yacc.c:1646  */
    { add_scalar_type_spec( SURFREF_TYPE ); }
#line 2395 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 387 "ptx.y" /* yacc.c:1646  */
    { add_array_initializer(); }
#line 2401 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 388 "ptx.y" /* yacc.c:1646  */
    { syntax_not_implemented(); }
#line 2407 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 394 "ptx.y" /* yacc.c:1646  */
    { add_label((yyvsp[-1].string_value)); }
#line 2413 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 397 "ptx.y" /* yacc.c:1646  */
    { set_return(); }
#line 2419 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 404 "ptx.y" /* yacc.c:1646  */
    { add_opcode((yyvsp[0].int_value)); }
#line 2425 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 405 "ptx.y" /* yacc.c:1646  */
    { add_opcode((yyvsp[0].int_value)); }
#line 2431 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 407 "ptx.y" /* yacc.c:1646  */
    { add_pred((yyvsp[0].string_value),0, -1); }
#line 2437 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 408 "ptx.y" /* yacc.c:1646  */
    { add_pred((yyvsp[0].string_value),1, -1); }
#line 2443 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 409 "ptx.y" /* yacc.c:1646  */
    { add_pred((yyvsp[-1].string_value),0,1); }
#line 2449 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 410 "ptx.y" /* yacc.c:1646  */
    { add_pred((yyvsp[-1].string_value),0,2); }
#line 2455 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 411 "ptx.y" /* yacc.c:1646  */
    { add_pred((yyvsp[-1].string_value),0,3); }
#line 2461 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 412 "ptx.y" /* yacc.c:1646  */
    { add_pred((yyvsp[-1].string_value),0,5); }
#line 2467 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 413 "ptx.y" /* yacc.c:1646  */
    { add_pred((yyvsp[-1].string_value),0,6); }
#line 2473 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 414 "ptx.y" /* yacc.c:1646  */
    { add_pred((yyvsp[-1].string_value),0,10); }
#line 2479 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 415 "ptx.y" /* yacc.c:1646  */
    { add_pred((yyvsp[-1].string_value),0,12); }
#line 2485 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 416 "ptx.y" /* yacc.c:1646  */
    { add_pred((yyvsp[-1].string_value),0,13); }
#line 2491 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 417 "ptx.y" /* yacc.c:1646  */
    { add_pred((yyvsp[-1].string_value),0,17); }
#line 2497 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 418 "ptx.y" /* yacc.c:1646  */
    { add_pred((yyvsp[-1].string_value),0,19); }
#line 2503 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 419 "ptx.y" /* yacc.c:1646  */
    { add_pred((yyvsp[-1].string_value),0,28); }
#line 2509 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 429 "ptx.y" /* yacc.c:1646  */
    { add_option(SYNC_OPTION); }
#line 2515 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 430 "ptx.y" /* yacc.c:1646  */
    { add_option(ARRIVE_OPTION); }
#line 2521 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 431 "ptx.y" /* yacc.c:1646  */
    { add_option(RED_OPTION); }
#line 2527 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 432 "ptx.y" /* yacc.c:1646  */
    { add_option(UNI_OPTION); }
#line 2533 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 433 "ptx.y" /* yacc.c:1646  */
    { add_option(WIDE_OPTION); }
#line 2539 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 434 "ptx.y" /* yacc.c:1646  */
    { add_option(ANY_OPTION); }
#line 2545 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 435 "ptx.y" /* yacc.c:1646  */
    { add_option(ALL_OPTION); }
#line 2551 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 436 "ptx.y" /* yacc.c:1646  */
    { add_option(BALLOT_OPTION); }
#line 2557 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 437 "ptx.y" /* yacc.c:1646  */
    { add_option(GLOBAL_OPTION); }
#line 2563 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 438 "ptx.y" /* yacc.c:1646  */
    { add_option(CTA_OPTION); }
#line 2569 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 439 "ptx.y" /* yacc.c:1646  */
    { add_option(SYS_OPTION); }
#line 2575 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 440 "ptx.y" /* yacc.c:1646  */
    { add_option(GEOM_MODIFIER_1D); }
#line 2581 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 441 "ptx.y" /* yacc.c:1646  */
    { add_option(GEOM_MODIFIER_2D); }
#line 2587 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 442 "ptx.y" /* yacc.c:1646  */
    { add_option(GEOM_MODIFIER_3D); }
#line 2593 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 443 "ptx.y" /* yacc.c:1646  */
    { add_option(SAT_OPTION); }
#line 2599 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 444 "ptx.y" /* yacc.c:1646  */
    { add_option(FTZ_OPTION); }
#line 2605 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 445 "ptx.y" /* yacc.c:1646  */
    { add_option(NEG_OPTION); }
#line 2611 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 446 "ptx.y" /* yacc.c:1646  */
    { add_option(APPROX_OPTION); }
#line 2617 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 447 "ptx.y" /* yacc.c:1646  */
    { add_option(FULL_OPTION); }
#line 2623 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 448 "ptx.y" /* yacc.c:1646  */
    { add_option(EXIT_OPTION); }
#line 2629 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 449 "ptx.y" /* yacc.c:1646  */
    { add_option(ABS_OPTION); }
#line 2635 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 451 "ptx.y" /* yacc.c:1646  */
    { add_option(TO_OPTION); }
#line 2641 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 452 "ptx.y" /* yacc.c:1646  */
    { add_option(HALF_OPTION); }
#line 2647 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 453 "ptx.y" /* yacc.c:1646  */
    { add_option(EXTP_OPTION); }
#line 2653 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 454 "ptx.y" /* yacc.c:1646  */
    { add_option(CA_OPTION); }
#line 2659 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 455 "ptx.y" /* yacc.c:1646  */
    { add_option(CG_OPTION); }
#line 2665 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 456 "ptx.y" /* yacc.c:1646  */
    { add_option(CS_OPTION); }
#line 2671 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 457 "ptx.y" /* yacc.c:1646  */
    { add_option(LU_OPTION); }
#line 2677 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 458 "ptx.y" /* yacc.c:1646  */
    { add_option(CV_OPTION); }
#line 2683 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 459 "ptx.y" /* yacc.c:1646  */
    { add_option(WB_OPTION); }
#line 2689 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 460 "ptx.y" /* yacc.c:1646  */
    { add_option(WT_OPTION); }
#line 2695 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 461 "ptx.y" /* yacc.c:1646  */
    { add_option(NC_OPTION); }
#line 2701 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 462 "ptx.y" /* yacc.c:1646  */
    { add_option(UP_OPTION); }
#line 2707 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 463 "ptx.y" /* yacc.c:1646  */
    { add_option(DOWN_OPTION); }
#line 2713 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 464 "ptx.y" /* yacc.c:1646  */
    { add_option(BFLY_OPTION); }
#line 2719 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 465 "ptx.y" /* yacc.c:1646  */
    { add_option(IDX_OPTION); }
#line 2725 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 468 "ptx.y" /* yacc.c:1646  */
    { add_option(ATOMIC_AND); }
#line 2731 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 469 "ptx.y" /* yacc.c:1646  */
    { add_option(ATOMIC_POPC); }
#line 2737 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 200:
#line 470 "ptx.y" /* yacc.c:1646  */
    { add_option(ATOMIC_OR); }
#line 2743 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 471 "ptx.y" /* yacc.c:1646  */
    { add_option(ATOMIC_XOR); }
#line 2749 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 472 "ptx.y" /* yacc.c:1646  */
    { add_option(ATOMIC_CAS); }
#line 2755 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 473 "ptx.y" /* yacc.c:1646  */
    { add_option(ATOMIC_EXCH); }
#line 2761 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 474 "ptx.y" /* yacc.c:1646  */
    { add_option(ATOMIC_ADD); }
#line 2767 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 475 "ptx.y" /* yacc.c:1646  */
    { add_option(ATOMIC_INC); }
#line 2773 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 476 "ptx.y" /* yacc.c:1646  */
    { add_option(ATOMIC_DEC); }
#line 2779 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 477 "ptx.y" /* yacc.c:1646  */
    { add_option(ATOMIC_MIN); }
#line 2785 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 478 "ptx.y" /* yacc.c:1646  */
    { add_option(ATOMIC_MAX); }
#line 2791 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 484 "ptx.y" /* yacc.c:1646  */
    { add_option(RN_OPTION); }
#line 2797 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 485 "ptx.y" /* yacc.c:1646  */
    { add_option(RZ_OPTION); }
#line 2803 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 486 "ptx.y" /* yacc.c:1646  */
    { add_option(RM_OPTION); }
#line 2809 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 487 "ptx.y" /* yacc.c:1646  */
    { add_option(RP_OPTION); }
#line 2815 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 490 "ptx.y" /* yacc.c:1646  */
    { add_option(RNI_OPTION); }
#line 2821 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 491 "ptx.y" /* yacc.c:1646  */
    { add_option(RZI_OPTION); }
#line 2827 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 492 "ptx.y" /* yacc.c:1646  */
    { add_option(RMI_OPTION); }
#line 2833 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 493 "ptx.y" /* yacc.c:1646  */
    { add_option(RPI_OPTION); }
#line 2839 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 496 "ptx.y" /* yacc.c:1646  */
    { add_option(EQ_OPTION); }
#line 2845 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 497 "ptx.y" /* yacc.c:1646  */
    { add_option(NE_OPTION); }
#line 2851 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 498 "ptx.y" /* yacc.c:1646  */
    { add_option(LT_OPTION); }
#line 2857 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 499 "ptx.y" /* yacc.c:1646  */
    { add_option(LE_OPTION); }
#line 2863 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 500 "ptx.y" /* yacc.c:1646  */
    { add_option(GT_OPTION); }
#line 2869 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 501 "ptx.y" /* yacc.c:1646  */
    { add_option(GE_OPTION); }
#line 2875 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 502 "ptx.y" /* yacc.c:1646  */
    { add_option(LO_OPTION); }
#line 2881 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 503 "ptx.y" /* yacc.c:1646  */
    { add_option(LS_OPTION); }
#line 2887 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 227:
#line 504 "ptx.y" /* yacc.c:1646  */
    { add_option(HI_OPTION); }
#line 2893 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 228:
#line 505 "ptx.y" /* yacc.c:1646  */
    { add_option(HS_OPTION); }
#line 2899 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 506 "ptx.y" /* yacc.c:1646  */
    { add_option(EQU_OPTION); }
#line 2905 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 507 "ptx.y" /* yacc.c:1646  */
    { add_option(NEU_OPTION); }
#line 2911 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 508 "ptx.y" /* yacc.c:1646  */
    { add_option(LTU_OPTION); }
#line 2917 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 509 "ptx.y" /* yacc.c:1646  */
    { add_option(LEU_OPTION); }
#line 2923 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 233:
#line 510 "ptx.y" /* yacc.c:1646  */
    { add_option(GTU_OPTION); }
#line 2929 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 511 "ptx.y" /* yacc.c:1646  */
    { add_option(GEU_OPTION); }
#line 2935 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 512 "ptx.y" /* yacc.c:1646  */
    { add_option(NUM_OPTION); }
#line 2941 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 513 "ptx.y" /* yacc.c:1646  */
    { add_option(NAN_OPTION); }
#line 2947 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 519 "ptx.y" /* yacc.c:1646  */
    { add_scalar_operand( (yyvsp[0].string_value) ); }
#line 2953 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 520 "ptx.y" /* yacc.c:1646  */
    { add_neg_pred_operand( (yyvsp[0].string_value) ); }
#line 2959 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 521 "ptx.y" /* yacc.c:1646  */
    { add_scalar_operand( (yyvsp[0].string_value) ); change_operand_neg(); }
#line 2965 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 526 "ptx.y" /* yacc.c:1646  */
    { change_operand_neg(); }
#line 2971 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 528 "ptx.y" /* yacc.c:1646  */
    { add_address_operand((yyvsp[-2].string_value),(yyvsp[0].int_value)); }
#line 2977 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 529 "ptx.y" /* yacc.c:1646  */
    { add_scalar_operand( (yyvsp[-1].string_value) ); change_operand_lohi(1);}
#line 2983 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 530 "ptx.y" /* yacc.c:1646  */
    { add_scalar_operand( (yyvsp[-1].string_value) ); change_operand_lohi(1); change_operand_neg();}
#line 2989 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 531 "ptx.y" /* yacc.c:1646  */
    { add_scalar_operand( (yyvsp[-1].string_value) ); change_operand_lohi(2);}
#line 2995 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 532 "ptx.y" /* yacc.c:1646  */
    { add_scalar_operand( (yyvsp[-1].string_value) ); change_operand_lohi(2); change_operand_neg();}
#line 3001 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 533 "ptx.y" /* yacc.c:1646  */
    { add_2vector_operand((yyvsp[-2].string_value),(yyvsp[0].string_value)); change_double_operand_type(-1);}
#line 3007 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 534 "ptx.y" /* yacc.c:1646  */
    { add_2vector_operand((yyvsp[-3].string_value),(yyvsp[-1].string_value)); change_double_operand_type(-1); change_operand_lohi(1);}
#line 3013 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 535 "ptx.y" /* yacc.c:1646  */
    { add_2vector_operand((yyvsp[-3].string_value),(yyvsp[-1].string_value)); change_double_operand_type(-1); change_operand_lohi(2);}
#line 3019 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 536 "ptx.y" /* yacc.c:1646  */
    { add_2vector_operand((yyvsp[-2].string_value),(yyvsp[0].string_value)); change_double_operand_type(-3);}
#line 3025 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 537 "ptx.y" /* yacc.c:1646  */
    { add_2vector_operand((yyvsp[-3].string_value),(yyvsp[-1].string_value)); change_double_operand_type(-3); change_operand_lohi(1);}
#line 3031 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 538 "ptx.y" /* yacc.c:1646  */
    { add_2vector_operand((yyvsp[-3].string_value),(yyvsp[-1].string_value)); change_double_operand_type(-3); change_operand_lohi(2);}
#line 3037 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 541 "ptx.y" /* yacc.c:1646  */
    { add_2vector_operand((yyvsp[-3].string_value),(yyvsp[-1].string_value)); }
#line 3043 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 542 "ptx.y" /* yacc.c:1646  */
    { add_3vector_operand((yyvsp[-5].string_value),(yyvsp[-3].string_value),(yyvsp[-1].string_value)); }
#line 3049 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 543 "ptx.y" /* yacc.c:1646  */
    { add_4vector_operand((yyvsp[-7].string_value),(yyvsp[-5].string_value),(yyvsp[-3].string_value),(yyvsp[-1].string_value)); }
#line 3055 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 544 "ptx.y" /* yacc.c:1646  */
    { add_1vector_operand((yyvsp[-1].string_value)); }
#line 3061 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 547 "ptx.y" /* yacc.c:1646  */
    { add_scalar_operand((yyvsp[-1].string_value)); }
#line 3067 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 552 "ptx.y" /* yacc.c:1646  */
    { add_builtin_operand((yyvsp[-1].int_value),(yyvsp[0].int_value)); }
#line 3073 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 553 "ptx.y" /* yacc.c:1646  */
    { add_builtin_operand((yyvsp[0].int_value),-1); }
#line 3079 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 556 "ptx.y" /* yacc.c:1646  */
    { add_memory_operand(); }
#line 3085 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 557 "ptx.y" /* yacc.c:1646  */
    { add_memory_operand(); change_memory_addr_space((yyvsp[-3].string_value)); }
#line 3091 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 558 "ptx.y" /* yacc.c:1646  */
    { change_memory_addr_space((yyvsp[-3].string_value)); }
#line 3097 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 559 "ptx.y" /* yacc.c:1646  */
    { change_memory_addr_space((yyvsp[-3].string_value)); add_memory_operand();}
#line 3103 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 560 "ptx.y" /* yacc.c:1646  */
    { change_operand_neg(); }
#line 3109 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 563 "ptx.y" /* yacc.c:1646  */
    { add_double_operand((yyvsp[-2].string_value),(yyvsp[0].string_value)); change_double_operand_type(1); }
#line 3115 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 564 "ptx.y" /* yacc.c:1646  */
    { add_double_operand((yyvsp[-3].string_value),(yyvsp[-1].string_value)); change_double_operand_type(1); change_operand_lohi(1); }
#line 3121 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 565 "ptx.y" /* yacc.c:1646  */
    { add_double_operand((yyvsp[-3].string_value),(yyvsp[-1].string_value)); change_double_operand_type(1); change_operand_lohi(2); }
#line 3127 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 566 "ptx.y" /* yacc.c:1646  */
    { add_double_operand((yyvsp[-3].string_value),(yyvsp[0].string_value)); change_double_operand_type(2); }
#line 3133 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 567 "ptx.y" /* yacc.c:1646  */
    { add_double_operand((yyvsp[-4].string_value),(yyvsp[-1].string_value)); change_double_operand_type(2); change_operand_lohi(1); }
#line 3139 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 568 "ptx.y" /* yacc.c:1646  */
    { add_double_operand((yyvsp[-4].string_value),(yyvsp[-1].string_value)); change_double_operand_type(2); change_operand_lohi(2); }
#line 3145 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 569 "ptx.y" /* yacc.c:1646  */
    { add_address_operand((yyvsp[-3].string_value),(yyvsp[0].int_value)); change_double_operand_type(3); }
#line 3151 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 572 "ptx.y" /* yacc.c:1646  */
    { add_literal_int((yyvsp[0].int_value)); }
#line 3157 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 573 "ptx.y" /* yacc.c:1646  */
    { add_literal_float((yyvsp[0].float_value)); }
#line 3163 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 574 "ptx.y" /* yacc.c:1646  */
    { add_literal_double((yyvsp[0].double_value)); }
#line 3169 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 577 "ptx.y" /* yacc.c:1646  */
    { add_address_operand((yyvsp[0].string_value),0); }
#line 3175 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 578 "ptx.y" /* yacc.c:1646  */
    { add_address_operand((yyvsp[-1].string_value),0); change_operand_lohi(1);}
#line 3181 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 579 "ptx.y" /* yacc.c:1646  */
    { add_address_operand((yyvsp[-1].string_value),0); change_operand_lohi(2); }
#line 3187 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 580 "ptx.y" /* yacc.c:1646  */
    { add_address_operand((yyvsp[-2].string_value),(yyvsp[0].int_value)); }
#line 3193 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 581 "ptx.y" /* yacc.c:1646  */
    { add_address_operand2((yyvsp[0].int_value)); }
#line 3199 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
    break;


#line 3203 "/home/clli/fxp/gpgpu-sim_def/build/gcc-4.8.4/cuda-7050/debug/cuda-sim/ptx.tab.c" /* yacc.c:1646  */
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
#line 584 "ptx.y" /* yacc.c:1906  */


extern int ptx_lineno;
extern const char *g_filename;

void syntax_not_implemented()
{
	printf("Parse error (%s:%u): this syntax is not (yet) implemented:\n",g_filename,ptx_lineno);
	ptx_error(NULL);
	abort();
}

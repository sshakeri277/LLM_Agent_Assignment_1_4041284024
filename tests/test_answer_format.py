import unittest

from agent.answer_format import (canonical, clean, extract_final_answer, extract_spec,
                                 normalize, to_ascii_digits)


class SpecTests(unittest.TestCase):
    def test_template_spec_and_fill(self):
        spec = extract_spec("علی و مینا روی هم ۱۹ سیب دارند. هر کدام چند سیب دارند؟\n"
                            "پاسخ را دقیقاً با فرمت زیر برگردانید:\nAli=<integer>, Mina=<integer>")
        self.assertEqual(spec.kind, "template")
        self.assertEqual(spec.placeholders, (("Ali", "integer"), ("Mina", "integer")))
        self.assertEqual(normalize("**Ali = 12 , Mina=۷**", spec), "Ali=12, Mina=7")
        self.assertEqual(normalize("FINAL ANSWER: Mina=7, Ali=12", spec), "Ali=12, Mina=7")
        self.assertIsNone(normalize("Ali=12", spec))

    def test_options_block(self):
        spec = extract_spec("رنگ جعبه چیست؟\nفقط یکی از این سه عبارت را برگردانید:\nRed\nGreen\nBlue")
        self.assertEqual(spec.kind, "options")
        self.assertEqual(spec.options, ("Red", "Green", "Blue"))
        self.assertEqual(normalize("FINAL ANSWER: `green`", spec), "Green")
        self.assertEqual(normalize("The box is Blue.", spec), "Blue")
        self.assertIsNone(normalize("Purple", spec))
        self.assertIn("Red | Green | Blue", spec.description)

    def test_options_with_equals_and_bullets(self):
        spec = extract_spec("وضعیت X و Y چیست؟\nدقیقاً یکی از گزینه‌های زیر را برگردانید:\n"
                            "- X=Yes, Y=No\n- X=No, Y=Yes")
        self.assertEqual(spec.options, ("X=Yes, Y=No", "X=No, Y=Yes"))
        self.assertEqual(normalize("x=no,  y=yes", spec), "X=No, Y=Yes")
        self.assertEqual(normalize("**X=Yes, Y=No**", spec), "X=Yes, Y=No")

    def test_inline_numeric_choices(self):
        spec = extract_spec("توپ در کدام سبد است؟\nفقط یکی از اعداد ۴، ۵ یا ۶ را برگردانید.")
        self.assertEqual(spec.kind, "options")
        self.assertEqual(spec.options, ("4", "5", "6"))
        self.assertEqual(normalize("۵", spec), "5")
        self.assertEqual(normalize("Basket 6", spec), "6")

    def test_integer(self):
        s = extract_spec("قیمت چقدر بوده است؟\nفقط عدد صحیح را برگردانید.")
        self.assertEqual(s.kind, "integer")
        self.assertEqual(normalize("۴۸۰", s), "480")
        self.assertEqual(normalize("480.0", s), "480")
        self.assertEqual(normalize("about 480 units", s), "480")
        self.assertEqual(normalize("1,250", s), "1250")
        self.assertEqual(normalize("-3", s), "-3")
        self.assertIsNone(normalize("between 3 and 5", s))
        self.assertIsNone(normalize("7.5", s))

    def test_digits(self):
        s = extract_spec("رمز چیست؟\nفقط کد چهاررقمی را برگردانید.")
        self.assertEqual(s.kind, "digits")
        self.assertEqual(s.digits, 4)
        self.assertEqual(normalize("Code: 0427", s), "0427")
        self.assertIsNone(normalize("427", s))
        s3 = extract_spec("فقط کد سه‌رقمی را برگردانید.")
        self.assertEqual((s3.kind, s3.digits), ("digits", 3))

    def test_letter(self):
        s = extract_spec("کدام کار اول است؟\nفقط حرف آن کار را برگردانید.")
        self.assertEqual(s.kind, "letter")
        self.assertEqual(normalize("Task **c**", s), "C")
        self.assertEqual(normalize("B", s), "B")

    def test_free_and_clean(self):
        s = extract_spec("پایتخت ایران کجاست؟")
        self.assertEqual(s.kind, "free")
        self.assertEqual(clean("پاسخ: «Tehran»."), "Tehran")
        self.assertEqual(clean(r"$\boxed{42}$"), "42")
        self.assertEqual(clean("**Final Answer:** `x`"), "x")
        self.assertEqual(normalize("  Tehran. ", s), "Tehran")

    def test_ascii_digits(self):
        self.assertEqual(to_ascii_digits("۱۲۳٤٥"), "12345")

    def test_canonical(self):
        self.assertEqual(canonical("Ali=12, Mina=7"), canonical("ali=12,mina=7"))
        self.assertEqual(canonical("480"), canonical("480.0"))
        self.assertNotEqual(canonical("352"), canonical("325"))

    def test_extract_final_answer(self):
        self.assertEqual(extract_final_answer("reasoning...\nFINAL ANSWER: B\n"), "B")
        self.assertEqual(extract_final_answer("**Final Answer:** 7"), "7")
        self.assertEqual(extract_final_answer("FINAL ANSWER: 3\nmore\nFINAL ANSWER: 5"), "5")
        self.assertEqual(extract_final_answer("so the result is\n\n42\n"), "42")
        self.assertEqual(extract_final_answer("پاسخ نهایی: 12"), "12")
        self.assertIsNone(extract_final_answer(""))
        self.assertIsNone(extract_final_answer("   \n"))


class InstructionRegionTests(unittest.TestCase):
    """Format cues must come from the answer instruction, not from the problem body (review I1/I2)."""

    def test_range_in_body_is_not_options(self):
        s = extract_spec("هر جعبه یکی از اعداد ۱ تا ۹ را روی خود دارد. مجموع دو جعبه ۱۰ است و یکی ۵ است. "
                         "دیگری چند است؟\nفقط عدد صحیح را برگردانید.")
        self.assertEqual(s.kind, "integer")

    def test_digits_word_in_body_but_integer_instruction(self):
        s = extract_spec("چند عدد سه‌رقمی با ارقام متمایز از مجموعه {۱,۲,۳} می‌توان ساخت؟\nفقط عدد صحیح را برگردانید.")
        self.assertEqual(s.kind, "integer")
        self.assertEqual(normalize("6", s), "6")

    def test_ascii_statements_not_options(self):
        s = extract_spec("دقیقاً یکی از گزاره‌های زیر درست است:\nA: x > 5\nB: x < 3\nC: x = 4\n"
                         "x یک عدد صحیح بین ۱ و ۶ است. x چند است؟\nفقط عدد صحیح را برگردانید.")
        self.assertEqual(s.kind, "integer")

    def test_letter_cue_in_body_but_integer_instruction(self):
        s = extract_spec("هر کارت یک حرف دارد و حرف آن کارت با رنگش مرتبط است. چند کارت قرمز است؟\n"
                         "فقط عدد صحیح را برگردانید.")
        self.assertEqual(s.kind, "integer")

    def test_template_inside_sentence(self):
        s = extract_spec("سن پدر و پسر چقدر است؟\nپاسخ را با فرمت Father=<integer>, Son=<integer> برگردانید.")
        self.assertEqual(s.kind, "template")
        self.assertEqual(s.template, "Father=<integer>, Son=<integer>")
        self.assertEqual(normalize("Father = 40, Son=12", s), "Father=40, Son=12")

    def test_template_key_not_swallowing_words(self):
        s = extract_spec("How old are they? answer as Father=<integer>, Son=<integer>")
        self.assertEqual(s.placeholders, (("Father", "integer"), ("Son", "integer")))

    def test_template_extra_placeholder_no_crash(self):
        s = extract_spec("Give: A=<integer>, B=<integer>, total <integer>")
        self.assertEqual(normalize("A=1, B=2", s), "A=1, B=2")

    def test_persian_template_keys(self):
        s = extract_spec("هر کدام چند سیب دارند؟\nپاسخ را با فرمت زیر برگردانید:\nعلی=<integer>، مینا=<integer>")
        self.assertEqual(s.kind, "template")
        self.assertEqual(normalize("علی = ۳، مینا = ۴", s), "علی=3، مینا=4")

    def test_negative_after_prefix_without_colon(self):
        s = extract_spec("دمای نهایی چند درجه است؟\nفقط عدد صحیح را برگردانید.")
        self.assertEqual(normalize("Answer -3", s), "-3")
        self.assertEqual(normalize("FINAL ANSWER: -3", s), "-3")


class ExplanationTailTests(unittest.TestCase):
    """M2: an answer followed by a short justification must still normalize."""

    INT = "چند جعبه وجود دارد؟\nفقط عدد صحیح را برگردانید."
    OPT = "رنگ جعبه چیست؟\nفقط یکی از این سه عبارت را برگردانید:\nRed\nGreen\nBlue"
    LETTER = "کدام کار اول است؟\nفقط حرف آن کار را برگردانید."

    def test_integer_with_parenthetical_reason(self):
        self.assertEqual(normalize("x = 480 (since 6*80 = 480)", extract_spec(self.INT)), "480")

    def test_integer_with_because_clause(self):
        self.assertEqual(normalize("480 because 6 × 80 = 480", extract_spec(self.INT)), "480")

    def test_integer_with_persian_reason(self):
        self.assertEqual(normalize("۴۸۰ زیرا ۶ × ۸۰ = ۴۸۰", extract_spec(self.INT)), "480")

    def test_integer_after_last_equals(self):
        self.assertEqual(normalize("6 * 80 = 480", extract_spec(self.INT)), "480")

    def test_letter_with_reason_naming_other_letters(self):
        self.assertEqual(normalize("E (since A must come before D)", extract_spec(self.LETTER)), "E")

    def test_option_with_reason_naming_other_option(self):
        self.assertEqual(normalize("Green (not Blue)", extract_spec(self.OPT)), "Green")

    def test_strictly_valid_answers_are_untouched(self):
        self.assertEqual(normalize("-3", extract_spec(self.INT)), "-3")
        self.assertIsNone(normalize("(3 or 4)", extract_spec(self.INT)))


class NumberKindTests(unittest.TestCase):
    """M2: "only the number" (without "integer") must accept a decimal answer."""

    def test_only_number_cue_is_number_kind(self):
        s = extract_spec("میانگین چند است؟\nفقط عدد را برگردانید.")
        self.assertEqual(s.kind, "number")
        self.assertEqual(normalize("12.5", s), "12.5")
        self.assertEqual(normalize("12.50 units", s), "12.5")
        self.assertEqual(normalize("480.0", s), "480")
        self.assertIsNone(normalize("12 or 13", s))

    def test_integer_cue_still_integer(self):
        self.assertEqual(extract_spec("چند است؟\nفقط عدد صحیح را برگردانید.").kind, "integer")
        self.assertEqual(extract_spec("How many? Return only the integer.").kind, "integer")


class ExactNumberTests(unittest.TestCase):
    """Numbers are parsed as decimal text, never through float (no precision loss)."""

    INT = "چند حالت ممکن است؟\nفقط عدد صحیح را برگردانید."
    NUM = "میانگین چند است؟\nفقط عدد را برگردانید."

    def test_large_integers_keep_every_digit(self):
        spec = extract_spec(self.INT)
        self.assertEqual(normalize("9007199254740993", spec), "9007199254740993")
        self.assertEqual(normalize("15511210043330985984000000", spec), "15511210043330985984000000")
        self.assertEqual(normalize("1,234,567,890,123,456,789", spec), "1234567890123456789")
        self.assertEqual(normalize("-9007199254740993", spec), "-9007199254740993")

    def test_integer_kind_still_accepts_integral_decimals_only(self):
        spec = extract_spec(self.INT)
        self.assertEqual(normalize("5.0", spec), "5")
        self.assertEqual(normalize("007", spec), "7")
        self.assertIsNone(normalize("12.5", spec))
        self.assertIsNone(normalize("9007199254740993.5", spec))

    def test_decimals_are_not_rounded(self):
        spec = extract_spec(self.NUM)
        self.assertEqual(normalize("3.14159265", spec), "3.14159265")
        self.assertEqual(normalize("0.0000001", spec), "0.0000001")
        self.assertEqual(normalize("-0.50", spec), "-0.5")
        self.assertEqual(normalize("12345678901234567.5", spec), "12345678901234567.5")

    def test_canonical_keeps_distinct_large_numbers_distinct(self):
        self.assertNotEqual(canonical("9007199254740993"), canonical("9007199254740992"))
        self.assertEqual(canonical("12.0"), canonical("12"))
        self.assertEqual(canonical("0.50"), canonical("0.5"))


if __name__ == "__main__":
    unittest.main()

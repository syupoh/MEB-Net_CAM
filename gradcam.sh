 # ! / u s r / b i n   b a s h 
 # ! / b i n / b a s h 
 
 i f   [   - z   " $ 1 "   ] 
     t h e n 
         g p u = " 0 " 
     e l s e 
         g p u = " $ { 1 } " 
 f i 
 
 i f   [   $ { g p u }   - e q   " 0 "   ] 
     t h e n 
 
         s e a r c h _ d i r = ' . / g r a d c a m / i m g s / m a r k e t   . / g r a d c a m / i m g s / d u k e m t m c ' 
 #           . / g r a d c a m / i m g s / i m g s   . / g r a d c a m / i m g s / d u k e m t m c   . / g r a d c a m / i m g s / m a r k e t ' 
 #         s e a r c h _ d i r = ' . / g r a d c a m / i m g s / m a r k e t   . / g r a d c a m / i m g s / i m g s ' 
 
         u n e t _ s e t = ' 
         . / l o g s / _ _ u n e t / _ u n e t _ 0 6 1 5 _ m t o d _ I D / u n e t _ m a r k e t 1 5 0 1 _ d u k e m t m c _ r e s n e t 5 0 _ F _ 0 . 0 0 1 _ 0 . 5 _ 0 . 0 0 0 0 _ 0 . 0 7 0 0 _ 2 0 2 1 - 0 6 - 1 6 T 0 4 : 0 7 / c h e c k p o i n t _ 7 9 . p t h . t a r 
         . / l o g s / _ _ u n e t / _ u n e t _ 0 6 1 5 _ m t o d _ I D / u n e t _ m a r k e t 1 5 0 1 _ d u k e m t m c _ d e n s e n e t _ F _ 0 . 0 0 1 _ 0 . 5 _ 0 . 0 0 0 0 _ 0 . 0 7 0 0 _ 2 0 2 1 - 0 6 - 1 5 T 1 4 : 4 2 / c h e c k p o i n t _ 7 9 . p t h . t a r 
         . / l o g s / _ _ u n e t / _ u n e t _ 0 6 1 5 _ m t o d _ I D / u n e t _ m a r k e t 1 5 0 1 _ d u k e m t m c _ i n c e p t i o n v 3 _ F _ 0 . 0 0 1 _ 0 . 5 _ 0 . 0 0 0 0 _ 0 . 0 7 0 0 _ 2 0 2 1 - 0 6 - 1 5 T 2 1 : 2 1 / c h e c k p o i n t _ 7 9 . p t h . t a r 
         ' 
 #     . / l o g s / _ _ u n e t / _ u n e t _ 0 6 1 5 _ m t o d / u n e t _ m a r k e t 1 5 0 1 _ d u k e m t m c _ r e s n e t 5 0 _ F _ 0 . 0 0 1 _ 0 . 5 _ 0 . 0 0 0 0 _ 0 . 7 0 0 0 _ 2 0 2 1 - 0 6 - 1 6 T 0 6 : 4 0 / c h e c k p o i n t _ 7 9 . p t h . t a r 
 
 
         f o r   i m a g e _ d i r   i n   $ { s e a r c h _ d i r } 
         d o 
             f o r   u n e t   i n   $ { u n e t _ s e t } 
             d o 
                 p y t h o n   g r a d c a m _ i d . p y   - - g p u = $ { g p u }   \ 
                       - - u n e t = $ { u n e t }   - - o n e h o t = 0   \ 
                       - - i m a g e - p a t h = $ { i m a g e _ d i r } / 
                 p y t h o n   g r a d c a m _ i d . p y   - - g p u = $ { g p u }   \ 
                       - - u n e t = $ { u n e t }   - - o n e h o t = 1   \ 
                       - - i m a g e - p a t h = $ { i m a g e _ d i r } / 
                 p y t h o n   g r a d c a m _ i d . p y   - - g p u = $ { g p u }   \ 
                       - - u n e t = $ { u n e t }   \ 
                       - - i m a g e - p a t h = $ { i m a g e _ d i r } / 
             d o n e 
         d o n e 
 # 
 #         S O U R C E = m a r k e t 1 5 0 1 
 #         T A R G E T = d u k e m t m c 
 # 
 # #         p y t h o n 3   m a i n / m i n e _ u n e t _ o n l y . p y   - d s   $ { T A R G E T }   - d t   $ { S O U R C E }     \ 
 # #           - - m a r g i n   0 . 0   - - n u m - i n s t a n c e   4   - b   1 6   - j   1   - - w a r m u p - s t e p   1 0   - - l r   0 . 0 0 0 3 5   - - l r 2   $ { l r 2 }   \ 
 # #           - - m i l e s t o n e s   4 0   7 0   - - e p o c h   8 0   - - e v a l - s t e p   5   - - g p u   $ { g p u }   - - a r c h   d e n s e n e t   \ 
 # #           - - a r c h - r e s u m e   $ { d e n s e _ p a t h } / m o d e l _ b e s t . p t h . t a r   \ 
 # #           - - l o g s - d i r   ' . / l o g s / _ _ u n e t / ' $ { u n e t d i r n a m e }   \ 
 # #           - - a l p h a   $ { a l p h a }   - - b e t a   $ { b e t a }   - - t y p e   $ { t y p e }   - - d e l t a   $ { d e l t a } 
 #         a t t n D _ s e t = ' . / l o g s / _ _ u n e t / _ u n e t _ 0 6 1 5 _ m t o d _ I D / u n e t _ m a r k e t 1 5 0 1 _ d u k e m t m c _ d e n s e n e t _ F _ 0 . 0 0 1 _ 0 . 5 _ 0 . 0 0 0 0 _ 0 . 0 7 0 0 _ 2 0 2 1 - 0 6 - 1 5 T 1 4 : 4 2 ' 
 #         f o r   a t t n D   i n   $ { a t t n D _ s e t } 
 #         d o 
 #             p y t h o n   m a i n / m i n e _ u n e t _ o n l y . p y   - d s   $ { T A R G E T }   - d t   $ { S O U R C E }   \ 
 #               - - g p u   $ { g p u }   - - a t t n D   $ { a t t n D } / c h e c k p o i n t _ 7 9 . p t h . t a r   - - p r i n t l y 
 #         d o n e 
 # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # 
         p a s s 
     e l i f   [   $ { g p u }   - e q   " 1 "   ] 
     t h e n 
 
         s e a r c h _ d i r = ' . / g r a d c a m / i m g s / m a r k e t   . / g r a d c a m / i m g s / i m g s   . / g r a d c a m / i m g s / d u k e m t m c ' 
 #         s e a r c h _ d i r = ' . / g r a d c a m / i m g s / m a r k e t   . / g r a d c a m / i m g s / i m g s ' 
 
         f o r   i m a g e _ d i r   i n   $ { s e a r c h _ d i r } 
         d o 
             p y t h o n   g r a d c a m _ i d . p y   - - g p u = $ { g p u }   \ 
                   - - i m a g e - p a t h = $ { i m a g e _ d i r } / 
         d o n e 
 
 
 # # # # # # # # # # # # # # # # # 
 # # # # # # # # # # # # # # # # # 
         p a s s 
     e l i f     [   $ { g p u }   - e q   " 2 "   ] 
     t h e n 
 
 
 #         S O U R C E = m a r k e t 1 5 0 1 
 #         T A R G E T = d u k e m t m c 
 # 
 # #         p y t h o n 3   m a i n / m i n e _ u n e t _ o n l y . p y   - d s   $ { T A R G E T }   - d t   $ { S O U R C E }     \ 
 # #           - - m a r g i n   0 . 0   - - n u m - i n s t a n c e   4   - b   1 6   - j   1   - - w a r m u p - s t e p   1 0   - - l r   0 . 0 0 0 3 5   - - l r 2   $ { l r 2 }   \ 
 # #           - - m i l e s t o n e s   4 0   7 0   - - e p o c h   8 0   - - e v a l - s t e p   5   - - g p u   $ { g p u }   - - a r c h   d e n s e n e t   \ 
 # #           - - a r c h - r e s u m e   $ { d e n s e _ p a t h } / m o d e l _ b e s t . p t h . t a r   \ 
 # #           - - l o g s - d i r   ' . / l o g s / _ _ u n e t / ' $ { u n e t d i r n a m e }   \ 
 # #           - - a l p h a   $ { a l p h a }   - - b e t a   $ { b e t a }   - - t y p e   $ { t y p e }   - - d e l t a   $ { d e l t a } 
 #         a t t n D _ s e t = ' . / l o g s / _ _ u n e t / _ u n e t _ 0 6 1 5 _ m t o d _ I D / u n e t _ m a r k e t 1 5 0 1 _ d u k e m t m c _ d e n s e n e t _ F _ 0 . 0 0 1 _ 0 . 5 _ 0 . 0 0 0 0 _ 0 . 0 7 0 0 _ 2 0 2 1 - 0 6 - 1 5 T 1 4 : 4 2 
 #           ' 
 # 
 #         f o r   a t t n D   i n   $ { a t t n D _ s e t } 
 #         d o 
 #             p y t h o n   m a i n / m i n e _ u n e t _ o n l y . p y   - d s   $ { T A R G E T }   - d t   $ { S O U R C E }   \ 
 #               - - g p u   $ { g p u }   - - a t t n D   $ { a t t n D } / c h e c k p o i n t _ 7 9 . p t h . t a r 
 #         d o n e 
 # 
 #         s e a r c h _ d i r = ' . / g r a d c a m / i m g s / m a r k e t   . / g r a d c a m / i m g s / i m g s   . / g r a d c a m / i m g s / d u k e m t m c ' 
 # #         s e a r c h _ d i r = ' . / g r a d c a m / i m g s / m a r k e t   . / g r a d c a m / i m g s / i m g s ' 
 # 
 #         f o r   i m a g e _ d i r   i n   $ { s e a r c h _ d i r } 
 #         d o 
 #             p y t h o n   g r a d c a m _ i d . p y   - - g p u = $ { g p u }   \ 
 #                   - - i m a g e - p a t h = $ { i m a g e _ d i r } / 
 #         d o n e 
 
 # # # # # # # # # # # # # # # # # 
         p a s s 
 f i 
 
 
 #         s e a r c h _ d i r = ' . / g r a d c a m / d u k e m t m c   . / g r a d c a m / m a r k e t ' 
 #         m o d e l _ d i r _ s e t = ' . / l o g s / _ 0 5 0 5 _ t y p e F ' 
 # 
 #         f o r   m o d e l _ d i r   i n   $ { m o d e l _ d i r _ s e t } 
 #         d o 
 #             f o r   m o d e l _ n a m e   i n   ` l s   $ { m o d e l _ d i r } ` 
 #             d o 
 #                 f o r   i m a g e _ d i r   i n   $ { s e a r c h _ d i r } 
 #                 d o 
 #                     f o r   i m a g e   i n   ` l s   $ { i m a g e _ d i r } ` 
 #                     d o 
 #                         p y t h o n   g r a d c a m _ d . p y   - - g p u = $ { g p u }   \ 
 #                           - - m o d e l = $ { m o d e l _ d i r } / $ { m o d e l _ n a m e } / c h e c k p o i n t _ 7 9 . p t h . t a r   - - i m a g e - p a t h = $ { i m a g e _ d i r } / $ { i m a g e } 
 #                     d o n e 
 #                 d o n e 
 #             d o n e 
 #         d o n e 
 
 
 # p y t h o n   g r a d c a m . p y   - - u s e - c u d a   - - i m a g e - p a t h   . / g r a d c a m / i m g s / p e r s o n 1 . j p g   - - m o d e l   . / g r a d c a m / r e s n e t _ m a r k e t . p t h . t a r   - - g p u   $ { g p u } 
 # p y t h o n   g r a d c a m . p y   - - u s e - c u d a   - - i m a g e - p a t h   . / g r a d c a m / i m g s / p e r s o n 1 . j p g   - - m o d e l   . / g r a d c a m / d e n s e n e t _ m a r k e t . p t h . t a r 
 # p y t h o n   g r a d c a m . p y   - - u s e - c u d a   - - i m a g e - p a t h   . / g r a d c a m / i m g s / p e r s o n 1 . j p g   - - m o d e l   . / g r a d c a m / i n c e p t i o n v 3 _ m a r k e t . p t h . t a r 
 
 # i m g _ s e t = ' p e r s o n 1 . j p g   p e r s o n 2 . j p g   p e r s o n 3 . j p g   p e r s o n 4 . j p g   p e r s o n 5 . j p g   p e r s o n 6 . j p g   p e r s o n 7 . j p g   p e r s o n 8 . j p g ' 
 # 
 # f o r   i m g   i n   $ { i m g _ s e t } 
 # d o 
 #     p y t h o n   g r a d c a m . p y   - - u s e - c u d a   - - i m a g e - p a t h   . / g r a d c a m / i m g s / $ { i m g }   - - m o d e l   . / g r a d c a m / r e s n e t _ m a r k e t . p t h . t a r 
 # d o n e 
 # 

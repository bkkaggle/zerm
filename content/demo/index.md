+++
title="demo"
description="a basic demo of zola."
date=2019-08-06

[taxonomies]
tags = ["demo", "zola", "highlighting"]
categories = ["programming", "wu tang",]

[extra]
+++

$$\sum$$

This is zerm, a minimalist theme for Zola based[^1] off of [panr's](https://twitter.com/panr)
theme for Hugo.

Inline code: `println!("Wu Tang!");`

```rs
fn foo(arg: String) -> Result<u32, Io::Error> {
    println!("Nice!"); // TODO: the thingy
    if 1 != 0 {
        println!("How many ligatures can I contrive??");
        println!("Turns out a lot! ==> -/-> <!-- <$> >>=");
    }
    Ok(42)
}
```

> In Hotel Rwanda, reminder to honor these street scholars who ask why
U.S. Defense is twenty percent of the tax dollar. Bush gave 6.46 billion to
Halliburton for troops support efforts in Iraq; meanwhile, the hood is hurting,
please believe that.

### Header III

| members        | age | notable album | to be messed with?     |
| -------------- | --- | ------------- | ---------------------- |
| GZA            | 52  | Liquid Swords | no                     |
| Inspectah Deck | 49  | CZARFACE      | `protect ya neck, boy` |


{{ youtube(id="UUpuz8IObcs") }}

[^1]: fork? port? a little bit of the former, more of the latter?

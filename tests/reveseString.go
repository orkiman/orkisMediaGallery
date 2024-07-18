package main

import (
	"fmt"
)

func main() {
	in := "א .   לכו ןידכ בכרב קיזחמה ,בכרה לעב ,הסילופה לעב קר  ימ תאמ רתיהב בכרב שמתשמה בכרב גוהנל םיאשר םהמ . ב .   גווסמה בכרל וא עונפואל האצוהש חוטיב תד"
	out := []rune(in)
	for s, e := 0, len(out)-1; s < e; s, e = s+1, e-1 {
		out[s], out[e] = out[e], out[s]
	}
	fmt.Println(string(out))
	fmt.Println(len(out))
	fmt.Println(in)
	fmt.Println(len(in))

}

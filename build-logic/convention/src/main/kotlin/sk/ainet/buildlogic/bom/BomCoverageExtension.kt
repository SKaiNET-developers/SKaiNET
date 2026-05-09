package sk.ainet.buildlogic.bom

import org.gradle.api.provider.SetProperty

abstract class BomCoverageExtension {
    abstract val excludePublished: SetProperty<String>
}
